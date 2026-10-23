#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-$(xcode-select -p)}"
mkdir -p validation build/test-module-cache
python3 scripts/validate_bundle.py > validation/checkout-integrity.log
xcrun clang++ -std=c++17 -F Vendor tests/SAMPromptTests.cpp -o validation/sam-prompt-tests
validation/sam-prompt-tests
xcrun clang++ -std=c++17 tests/SAMInferenceTimingTests.cpp -o validation/sam-inference-timing-tests
validation/sam-inference-timing-tests
xcrun swiftc -module-cache-path build/test-module-cache Sources/CSVLog.swift tests/CSVLogTests.swift -o validation/csv-tests
validation/csv-tests validation/csv-fixtures
python3 scripts/check_csv_test.py
xcodebuild -project RuntimeTest.xcodeproj -scheme RuntimeTest -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO build > validation/build-release.log 2>&1
python3 scripts/validate_bundle.py --app build/Build/Products/Release-iphoneos/RuntimeTest.app
echo 'PASS: unsigned arm64 iPhone Release build and bundled data/models'
