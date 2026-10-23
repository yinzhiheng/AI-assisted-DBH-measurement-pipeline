import SwiftUI
import UIKit

struct DatasetManifest: Decodable {
    let commit: String
    let records: [DatasetImage]
    let totalBytes: Int
}

struct DatasetImage: Decodable, Identifiable, Hashable {
    let id: String
    let field: String
    let rgb: String
    let depth: String
    let baseX: Int
    let baseY: Int
    let rgbSHA1: String
    let depthSHA1: String
    var title: String { "\(field == "natural" ? "Natural" : "Plotted") · \(id.prefix(8))" }
}

enum BenchmarkError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case let .message(text) = self { return text }; return nil }
}

// One engine on one serial queue: neither Ort sessions nor the copied C++ globals
// are accessed concurrently. The main thread only handles UI and CSV appends.
final class BenchmarkWorker: @unchecked Sendable {
    private let queue = DispatchQueue(label: "runTimeTest.inference", qos: .userInitiated)
    private let engine = RuntimeBridge()

    func load() async -> String? {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume(returning: self.engine.loadModels()) }
        }
    }
    func run(_ item: DatasetImage, root: URL, preview: Bool) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            queue.async {
                let result: [String: Any] = autoreleasepool {
                    self.engine.runRGB(root.appendingPathComponent(item.rgb).path,
                                       depth: root.appendingPathComponent(item.depth).path,
                                       baseX: item.baseX, baseY: item.baseY, preview: preview)
                }
                continuation.resume(returning: result)
            }
        }
    }
}

@MainActor
final class BenchmarkModel: ObservableObject {
    static let batchPauseMilliseconds: UInt64 = 200

    @Published var images: [DatasetImage] = []
    @Published var selected: DatasetImage?
    @Published var preview: UIImage?
    @Published var busy = false
    @Published var stopping = false
    @Published var completed = 0
    @Published var total = 0
    @Published var successful = 0
    @Published var message = "Loading bundled dataset…"
    @Published var latestMS: Double?
    @Published var csvFiles: [URL] = []
    @Published var currentCSV: URL?
    @Published var datasetReady = false
    private let worker = BenchmarkWorker()
    private var datasetRoot: URL?
    private var cancelRequested = false
    private let resultsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Results", isDirectory: true)

    init() {
        refreshCSVFiles()
        do {
            guard let root = Bundle.main.resourceURL?.appendingPathComponent("Dataset"),
                  FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path) else {
                throw BenchmarkError.message("Dataset missing. Run scripts/prepare_dataset.py on your Mac, then rebuild/install the app.")
            }
            let manifest = try JSONDecoder().decode(DatasetManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
            guard manifest.records.count == 487, Set(manifest.records.map(\.id)).count == 487 else {
                throw BenchmarkError.message("Dataset manifest is incomplete or contains duplicate image IDs.")
            }
            for item in manifest.records {
                guard FileManager.default.fileExists(atPath: root.appendingPathComponent(item.rgb).path),
                      FileManager.default.fileExists(atPath: root.appendingPathComponent(item.depth).path) else {
                    throw BenchmarkError.message("Missing RGB/depth pair: \(item.id). Re-run dataset preparation and reinstall.")
                }
            }
            images = manifest.records; datasetRoot = root
            datasetReady = true
            message = "487 images ready offline · 285 natural + 202 plotted"
            select(images[0])
        } catch { message = error.localizedDescription }
    }

    func select(_ item: DatasetImage) {
        guard !busy else { return }
        selected = item; latestMS = nil
        if let root = datasetRoot { preview = UIImage(contentsOfFile: root.appendingPathComponent(item.rgb).path) }
    }
    func stop() {
        guard busy else { return }
        cancelRequested = true; stopping = true
        message = "Stopping after the current image; completed CSV rows are saved."
    }
    func refreshCSVFiles() {
        csvFiles = ((try? FileManager.default.contentsOfDirectory(at: resultsDirectory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "csv" }.sorted { $0.lastPathComponent > $1.lastPathComponent }
    }
    func start(batch: Bool) {
        guard !busy, datasetReady, let root = datasetRoot else { return }
        let items = batch ? images : selected.map { [$0] } ?? []
        guard !items.isEmpty else { return }
        #if targetEnvironment(simulator)
        message = "This benchmark must run on a physical iPhone. Select your iPhone in Xcode."
        return
        #else
        busy = true; stopping = false; cancelRequested = false
        completed = 0; successful = 0; total = items.count; latestMS = nil; currentCSV = nil
        if !batch { preview = UIImage(contentsOfFile: root.appendingPathComponent(items[0].rgb).path) }
        UIApplication.shared.isIdleTimerDisabled = true
        message = "Loading MobileSAMv2 (outside the timer)…"
        Task {
            defer {
                busy = false; stopping = false
                UIApplication.shared.isIdleTimerDisabled = false
                refreshCSVFiles()
            }
            do {
                if let failure = await worker.load() { throw BenchmarkError.message(failure) }
                if cancelRequested { message = "Stopped before inference."; return }
                let runID = UUID().uuidString
                let mode = batch ? "batch" : "single"
                let log = try CSVLog(directory: resultsDirectory, runID: runID, mode: mode)
                currentCSV = log.url
                for (index, item) in items.enumerated() {
                    if cancelRequested { break }
                    if batch && index > 0 {
                        // Pause between images, outside inference timing. Suspending
                        // this task keeps the UI and Stop button responsive.
                        try await Task.sleep(nanoseconds: Self.batchPauseMilliseconds * 1_000_000)
                        if cancelRequested { break }
                    }
                    message = "\(index + 1)/\(items.count) · \(item.title)"
                    let result = await worker.run(item, root: root, preview: !batch)
                    latestMS = result["segmentation_ms"] as? Double
                    try log.append(imageName: URL(fileURLWithPath: item.rgb).lastPathComponent,
                                   segmentationMilliseconds: latestMS)
                    completed += 1
                    if result["status"] as? String == "ok" { successful += 1 }
                    if let image = result["preview"] as? UIImage { preview = image }
                    if !batch {
                        let timing = latestMS.map { String(format: "%.1f ms", $0) } ?? "not timed"
                        message = "\(result["status"] as? String ?? "unknown"): \(timing)\n\(result["error"] as? String ?? "")"
                    }
                }
                if batch || cancelRequested {
                    message = "\(cancelRequested ? "Stopped" : "Finished"): \(completed)/\(total) images, \(successful) successful. CSV saved."
                }
            } catch {
                message = "Stopped: \(error.localizedDescription) Completed rows remain in Results."
            }
        }
        #endif
    }
}
