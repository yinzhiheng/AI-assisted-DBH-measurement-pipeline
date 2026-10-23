import SwiftUI

@main
struct RuntimeTestApp: App {
    @StateObject private var model = BenchmarkModel()
    var body: some Scene {
        WindowGroup { BenchmarkView(model: model) }
    }
}

private struct SharedFile: Identifiable {
    let url: URL
    var id: String { url.path }
}

struct BenchmarkView: View {
    @ObservedObject var model: BenchmarkModel
    @State private var choosingImage = false
    @State private var showingResults = false
    @State private var sharedFile: SharedFile?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("MobileSAMv2").font(.largeTitle.bold())
                        Text("iForester · on-device segmentation benchmark")
                            .font(.subheadline).foregroundColor(.secondary)
                    }
                    if let image = model.preview {
                        Image(uiImage: image).resizable().scaledToFit()
                            .frame(maxWidth: .infinity).frame(height: 280)
                            .background(Color.black.opacity(0.04)).cornerRadius(12)
                    }
                    if let selected = model.selected {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(selected.title).font(.headline)
                            Text(selected.id).font(.caption2.monospaced()).foregroundColor(.secondary)
                        }
                    }
                    Button { choosingImage = true } label: {
                        Label("Select image", systemImage: "photo.on.rectangle").frame(maxWidth: .infinity)
                    }.buttonStyle(.bordered).disabled(model.busy || !model.datasetReady)
                    HStack(spacing: 10) {
                        Button { model.start(batch: false) } label: {
                            Label("Segment selected", systemImage: "play.fill").frame(maxWidth: .infinity)
                        }.disabled(model.busy || model.selected == nil || !model.datasetReady)
                        Button { model.start(batch: true) } label: {
                            Label("Batch all 487", systemImage: "square.stack.3d.up").frame(maxWidth: .infinity)
                        }.disabled(model.busy || !model.datasetReady)
                    }.buttonStyle(.borderedProminent).controlSize(.large)
                    Text("Batch pauses \(BenchmarkModel.batchPauseMilliseconds) ms between images (excluded from timing).")
                        .font(.caption).foregroundColor(.secondary)
                    if model.busy {
                        ProgressView(value: Double(model.completed), total: Double(max(1, model.total)))
                        Button(model.stopping ? "Stopping…" : "Stop after current image", role: .destructive) { model.stop() }
                            .disabled(model.stopping)
                    }
                    if let ms = model.latestMS {
                        Text(String(format: "%.1f ms", ms)).font(.system(size: 34, weight: .semibold, design: .rounded))
                    }
                    Text(model.message).font(.callout).textSelection(.enabled)
                    VStack(alignment: .leading, spacing: 7) {
                        Text("What is timed").font(.headline)
                        Text("Fresh image encoding + box-prompt mask decoding only.")
                    }.font(.caption).foregroundColor(.secondary)
                    Button { if let url = model.currentCSV { sharedFile = SharedFile(url: url) } } label: {
                        Label("Export latest CSV", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity)
                    }.buttonStyle(.bordered).disabled(model.currentCSV == nil || model.busy)
                    Button("Saved CSV files (\(model.csvFiles.count))") { model.refreshCSVFiles(); showingResults = true }
                        .disabled(model.busy)
                    Text("Files → On My iPhone → Runtime Test → Results")
                        .font(.caption).foregroundColor(.secondary)
                }.padding(20)
            }
            .navigationTitle("Runtime Test").navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
        .sheet(isPresented: $choosingImage) { ImagePickerView(model: model) }
        .sheet(isPresented: $showingResults) {
            NavigationView {
                List(model.csvFiles, id: \.path) { url in
                    Button(url.lastPathComponent) {
                        showingResults = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { sharedFile = SharedFile(url: url) }
                    }.font(.caption)
                }.navigationTitle("Saved CSV files")
                    .toolbar { Button("Done") { showingResults = false } }
            }
        }
        .sheet(item: $sharedFile) { ShareSheet(url: $0.url) }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in model.stop() }
    }
}

private struct ImagePickerView: View {
    @ObservedObject var model: BenchmarkModel
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""
    var body: some View {
        NavigationView {
            List {
                ForEach(["natural", "plotted"], id: \.self) { field in
                    Section(field == "natural" ? "Natural field · 285" : "Plotted field · 202") {
                        ForEach(model.images.filter { $0.field == field && (search.isEmpty || $0.id.localizedCaseInsensitiveContains(search)) }) { item in
                            Button { model.select(item); dismiss() } label: {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(item.title)
                                        Text(item.id).font(.caption2).foregroundColor(.secondary)
                                    }
                                    if model.selected == item { Spacer(); Image(systemName: "checkmark") }
                                }
                            }
                        }
                    }
                }
            }.searchable(text: $search, prompt: "Search image ID")
                .navigationTitle("Select an image")
                .toolbar { Button("Done") { dismiss() } }
        }
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
