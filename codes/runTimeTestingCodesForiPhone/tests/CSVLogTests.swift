import Foundation

@main
struct CSVLogTests {
    static func main() throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1])
        for mode in ["single", "batch"] {
            var log: CSVLog? = try CSVLog(directory: directory, runID: "test-run", mode: mode)
            let url = log!.url
            try log!.append(imageName: "rgb-comma, quote \" and\nnewline.jpeg", segmentationMilliseconds: nil)
            // Check that a completed row is readable before closing the writer.
            let first = try String(contentsOf: url, encoding: .utf8)
            precondition(first.contains("\"rgb-comma, quote \"\" and\nnewline.jpeg\""))
            try log!.append(imageName: "rgb-example.jpeg", segmentationMilliseconds: 123.456789)
            log = nil
            let closed = try String(contentsOf: url, encoding: .utf8)
            precondition(closed.hasPrefix("segmentation_ms,image_name\r\n"))
            precondition(closed.contains("123.456789"))
            print(url.path)
        }
    }
}
