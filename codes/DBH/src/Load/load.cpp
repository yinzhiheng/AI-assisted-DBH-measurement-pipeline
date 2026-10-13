#include "load.h"
#include <fstream>
#include <iterator>

cv::Point2i g_tap_full(-1, -1);
bool g_hasTap = false;
float g_displayScale = 1.0f;


// ------------------------- mouse callback -------------------------

void OnMouseRGB(int event, int x, int y, int, void*)
{
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    const int fx = (int)std::lround(x / g_displayScale);
    const int fy = (int)std::lround(y / g_displayScale);

    g_tap_full = cv::Point2i(fx, fy);
    g_hasTap = true;

    std::cout << "Tap selected (full-res): (" << fx << ", " << fy << ")\n";
}

cv::Mat readImageFile(const std::filesystem::path& path, int flags)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open image: " + path.u8string());
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),
                                           std::istreambuf_iterator<char>());
    if (input.bad() || bytes.empty())
        throw std::runtime_error("Cannot read image: " + path.u8string());
    cv::Mat image = cv::imdecode(bytes, flags);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + path.u8string());
    return image;
}

cv::Mat readDepthMap(const std::string& combinePngPath)
{
    cv::Mat image = cv::imread(combinePngPath, cv::IMREAD_UNCHANGED);
    if (image.empty())
        throw std::runtime_error("Failed to read combine PNG: " + combinePngPath);

    return readDepthMap(image);
}

cv::Mat readDepthMap(const cv::Mat& image)
{
    if (image.rows <= 4)
        throw std::runtime_error("Combine PNG has too few rows (<=4)");

    cv::Mat rawGray;
    if (image.channels() == 1)
    {
        rawGray = image;
    }
    else
    {
        std::vector<cv::Mat> ch;
        cv::split(image, ch);
        rawGray = ch[0];
    }

    if (rawGray.depth() != CV_16U && rawGray.depth() != CV_16S)
    {
        throw std::runtime_error(
            "Unexpected combine PNG depth type. Expect 16-bit. type=" + std::to_string(rawGray.type()));
    }

    cv::Mat depthPacked = rawGray.rowRange(4, rawGray.rows);

    cv::Mat depth_m;
    depthPacked.convertTo(depth_m, CV_32FC1, 6.0 / (255.0 * 255.0));
    return depth_m; // CV_32FC1, meters
}

// ------------------------- NEW: extract intrinsicString from PNG metadata rows -------------------------

std::string BytesToPrintablePreserveDelims(const std::vector<unsigned char>& bytes)
{
    std::string s;
    s.reserve(bytes.size());

    for (unsigned char b : bytes)
    {
        // keep key separators / punctuation
        if (b == '|' || b == ',' || b == '.' || b == '-' || b == '+')
        {
            s.push_back((char)b);
            continue;
        }

        // digits / letters / common symbols
        if ((b >= '0' && b <= '9') ||
            (b >= 'A' && b <= 'Z') ||
            (b >= 'a' && b <= 'z'))
        {
            s.push_back((char)b);
            continue;
        }

        // treat everything else as a separator
        s.push_back(' ');
    }

    return s;
}

std::vector<std::string> SplitByPipe(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == '|')
        {
            out.push_back(Trim(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(Trim(cur));

    // remove empty tokens
    std::vector<std::string> filtered;
    filtered.reserve(out.size());
    for (auto& t : out)
    {
        t = Trim(t);
        if (!t.empty()) filtered.push_back(t);
    }
    return filtered;
}


// Extract a quoted string value after a key occurrence: key ... ":" "VALUE"
bool ExtractQuotedValueAfterKey(const std::string& text, const std::string& key, std::string& valueOut)
{
    size_t kpos = text.find(key);
    if (kpos == std::string::npos) return false;

    // Search forward for the first quote after the key
    size_t q1 = text.find('"', kpos + key.size());
    if (q1 == std::string::npos) return false;

    // Sometimes it is key":"VALUE" so q1 might be the quote of VALUE or key,
    // so ensure we move to the quote that starts the value.
    // Find next quote after q1
    size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;

    // Heuristic: if the substring between q1 and q2 looks like ":" or key name, advance
    // We try to locate the quote that starts the actual value by looking for ':' then '"'
    size_t colon = text.find(':', kpos);
    if (colon != std::string::npos)
    {
        size_t vq1 = text.find('"', colon);
        if (vq1 != std::string::npos)
        {
            size_t vq2 = text.find('"', vq1 + 1);
            if (vq2 != std::string::npos && vq2 > vq1)
            {
                valueOut = Trim(text.substr(vq1 + 1, vq2 - (vq1 + 1)));
                return !valueOut.empty();
            }
        }
    }

    // fallback: take q1..q2
    valueOut = Trim(text.substr(q1 + 1, q2 - (q1 + 1)));
    return !valueOut.empty();
}

std::string RemoveAllSpaces(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
            out.push_back(c);
    }
    return out;
}

std::string readIntrinsicStringFromCombinePNG(const std::string& combinePngPath)
{
    cv::Mat image = cv::imread(combinePngPath, cv::IMREAD_UNCHANGED);
    if (image.empty())
        throw std::runtime_error("Failed to read combine PNG: " + combinePngPath);

    return readIntrinsicStringFromCombinePNG(image);
}

std::string readIntrinsicStringFromCombinePNG(const cv::Mat& image)
{
    if (image.rows < 4)
        throw std::runtime_error("Combine PNG has fewer than 4 rows");

    cv::Mat rawGray;
    if (image.channels() == 1) rawGray = image;
    else
    {
        std::vector<cv::Mat> ch;
        cv::split(image, ch);
        rawGray = ch[0];
    }

    if (rawGray.depth() != CV_16U && rawGray.depth() != CV_16S)
        throw std::runtime_error("Metadata decode expects 16-bit channel. type=" + std::to_string(rawGray.type()));

    // First 4 rows contain metadata packed as text
    cv::Mat meta = rawGray.rowRange(0, 4);

    // Convert uint16 stream into bytes (try both byte orders)
    std::vector<unsigned char> bytesLE;
    std::vector<unsigned char> bytesBE;
    bytesLE.reserve(meta.total() * 2);
    bytesBE.reserve(meta.total() * 2);

    for (int y = 0; y < meta.rows; ++y)
    {
        const uint16_t* row = meta.ptr<uint16_t>(y);
        for (int x = 0; x < meta.cols; ++x)
        {
            uint16_t v = row[x];
            unsigned char lo = (unsigned char)(v & 0xFF);
            unsigned char hi = (unsigned char)((v >> 8) & 0xFF);

            // little-endian
            bytesLE.push_back(lo);
            bytesLE.push_back(hi);

            // big-endian
            bytesBE.push_back(hi);
            bytesBE.push_back(lo);
        }
    }

    // Decode and split by '|'
    const std::string textLE = BytesToPrintablePreserveDelims(bytesLE);
    const std::string textBE = BytesToPrintablePreserveDelims(bytesBE);

    auto toksLE = SplitByPipe(textLE);
    auto toksBE = SplitByPipe(textBE);

    // Choose the one that looks more complete (more tokens)
    const auto& toks = (toksBE.size() > toksLE.size()) ? toksBE : toksLE;

    // We expect at least 21 metadata fields:
    // [0] version ... [18] moisture, [19] intrinsicString, [20] extrinsicString
    if (toks.size() < 21)
    {
        std::cerr << "[DEBUG] Not enough metadata tokens parsed. tokens=" << toks.size() << "\n";
        std::cerr << "[DEBUG] Preview tokens:\n";
        for (size_t i = 0; i < toks.size() && i < 10; ++i)
            std::cerr << "  [" << i << "] " << toks[i] << "\n";
        throw std::runtime_error("Failed to parse enough metadata fields from PNG header.");
    }

    //const std::string intrinsicString = toks[19];
    std::string intrinsicString = toks[19];
    intrinsicString = RemoveAllSpaces(intrinsicString);
    return intrinsicString;
}

// ------------------------- K from intrinsicString (matches your Python) -------------------------

cv::Mat readIntrinsicMatrixFromString(const std::string& intrinsicString, float scale)
{
    std::vector<float> values;
    values.reserve(9);

    std::stringstream ss(intrinsicString);
    std::string token;
    while (std::getline(ss, token, ','))
        values.push_back(std::stof(Trim(token)));

    if (values.size() != 9)
        throw std::runtime_error("Expected 9 values in intrinsicString, got " + std::to_string(values.size()));

    cv::Mat K_raw(3, 3, CV_32F);
    for (int i = 0; i < 9; ++i)
        K_raw.at<float>(i / 3, i % 3) = values[i];

    cv::Mat K = K_raw.t();
    K /= scale;

    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);
    K.at<float>(0, 2) = cy;
    K.at<float>(1, 2) = cx;
    K.at<float>(2, 2) = 1.0f;

    return K;
}

// ------------------------- small helpers -------------------------

std::string JoinPath(const std::string& dir, const std::string& file)
{
    if (dir.empty()) return file;
    const char last = dir.back();
    if (last == '\\' || last == '/') return dir + file;
    return dir + "\\" + file; // Windows-friendly
}

std::string Trim(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, e - b + 1);
}

void DrawPointLabel(cv::Mat& img, const cv::Point2i& p,
    const std::string& label, const cv::Scalar& color)
{
    cv::circle(img, p, 6, color, 2);
    cv::putText(img, label, p + cv::Point2i(10, -10),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
}

cv::Point2i FullToDepth(const cv::Point2i& p_full, float s)
{
    return cv::Point2i((int)std::lround(p_full.x / s), (int)std::lround(p_full.y / s));
}

cv::Mat VisualizeDepth(const cv::Mat& depth_m, float clipMin, float clipMax)
{
    cv::Mat d = depth_m.clone();
    for (int y = 0; y < d.rows; ++y)
    {
        float* row = d.ptr<float>(y);
        for (int x = 0; x < d.cols; ++x)
        {
            float v = row[x];
            if (!std::isfinite(v) || v <= 0) v = clipMax;
            row[x] = std::min(std::max(v, clipMin), clipMax);
        }
    }

    cv::Mat gray8;
    cv::Mat norm = (clipMax - d) / (clipMax - clipMin);
    norm.convertTo(gray8, CV_8U, 255.0);

    cv::Mat color;
    cv::applyColorMap(gray8, color, cv::COLORMAP_TURBO);
    return color;
}

void DrawDirectionArrow(
    cv::Mat& img,
    const cv::Point2f& origin,
    const cv::Vec2f& dir,          // should be unit or near-unit
    float length_px,
    const cv::Scalar& color,
    int thickness
) {
    const float n = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
    if (n < 1e-6f) return;

    cv::Point2f d(dir[0] / n, dir[1] / n);
    cv::Point2f tip = origin + d * length_px;

    cv::arrowedLine(
        img,
        cv::Point((int)std::lround(origin.x), (int)std::lround(origin.y)),
        cv::Point((int)std::lround(tip.x), (int)std::lround(tip.y)),
        color,
        thickness,
        cv::LINE_AA,
        0,
        0.25
    );
}
