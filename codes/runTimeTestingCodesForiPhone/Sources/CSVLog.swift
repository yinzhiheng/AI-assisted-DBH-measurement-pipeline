import Foundation

final class CSVLog {
    static let columns = ["segmentation_ms", "image_name"]
    let url: URL
    private let handle: FileHandle
    init(directory: URL, runID: String, mode: String) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        url = directory.appendingPathComponent("\(stamp)_\(mode)_\(runID.prefix(8)).csv")
        try (Self.columns.joined(separator: ",") + "\r\n").write(to: url, atomically: true, encoding: .utf8)
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }
    func append(imageName: String, segmentationMilliseconds: Double?) throws {
        let time = segmentationMilliseconds.map {
            String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), $0)
        } ?? ""
        let line = [time, imageName].map { value in
            "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }.joined(separator: ",") + "\r\n"
        try handle.write(contentsOf: Data(line.utf8))
        try handle.synchronize() // Each completed image survives an interrupted batch.
    }
    deinit { try? handle.close() }
}

