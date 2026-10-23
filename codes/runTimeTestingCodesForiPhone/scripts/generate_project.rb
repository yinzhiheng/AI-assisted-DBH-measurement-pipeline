#!/usr/bin/env ruby
# Rebuild only the .xcodeproj; source files and the dataset are never overwritten.
require 'xcodeproj'
root = File.expand_path('..', __dir__)
Dir.chdir(root)
project = Xcodeproj::Project.new('RuntimeTest.xcodeproj')
target = project.new_target(:application, 'RuntimeTest', :ios, '15.6')
sources = project.main_group.new_group('Sources')
vendor = project.main_group.new_group('Vendor')
resources = project.main_group.new_group('Resources')
Dir['Sources/*.{swift,mm,cpp,h,hpp}', 'Vendor/iForester/**/*.{cpp,h}', 'Vendor/SegmentAnything/*.{cpp,hpp}'].sort.each do |path|
  group = path.start_with?('Sources/') ? sources : vendor
  ref = group.new_file(path)
  target.source_build_phase.add_file_reference(ref) if %w[.swift .mm .cpp].include?(File.extname(path))
end
%w[opencv2 onnxruntime].each do |name|
  ref = vendor.new_file("Vendor/#{name}.framework")
  target.frameworks_build_phase.add_file_reference(ref)
end
# Both frameworks are static archives; they must be linked, never embedded.
%w[UIKit SwiftUI Foundation CoreGraphics CoreFoundation CoreMedia CoreVideo AVFoundation Accelerate AssetsLibrary QuartzCore ImageIO CoreML].each do |name|
  target.add_system_framework(name)
end
%w[mobile_sam.onnx mobile_sam_preprocess.onnx provenance.json MobileSAM-LICENSE.txt].each do |name|
  ref = resources.new_file("Vendor/SegmentAnything/mobile_sam/#{name}")
  target.resources_build_phase.add_file_reference(ref)
end
%w[source-provenance.json THIRD_PARTY_NOTICES.txt].each do |name|
  ref = resources.new_file("Resources/#{name}")
  target.resources_build_phase.add_file_reference(ref)
end
dataset = resources.new_file('Resources/Dataset')
dataset.last_known_file_type = 'folder'
target.resources_build_phase.add_file_reference(dataset)
project.main_group.new_file('Info.plist')
headers = ['$(inherited)', '$(SRCROOT)/Sources', '$(SRCROOT)/Vendor/SegmentAnything',
           '$(SRCROOT)/Vendor/onnxruntime.framework/Headers'] + Dir['Vendor/iForester/*'].map { |p| "$(SRCROOT)/#{p}" }
target.build_configurations.each do |config|
  s = config.build_settings
  s['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.yzhiheng.iForesterRuntimeTest'
  s['PRODUCT_NAME'] = 'RuntimeTest'
  s['INFOPLIST_FILE'] = 'Info.plist'
  s['SWIFT_VERSION'] = '5.0'
  s['SWIFT_OBJC_BRIDGING_HEADER'] = 'Sources/RuntimeTest-Bridging-Header.h'
  s['HEADER_SEARCH_PATHS'] = headers
  s['FRAMEWORK_SEARCH_PATHS'] = ['$(inherited)', '$(SRCROOT)/Vendor']
  s['OTHER_LDFLAGS'] = ['$(inherited)', '-lc++']
  s['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++17'
  s['CLANG_ENABLE_MODULES'] = 'YES'
  s['CLANG_ENABLE_OBJC_ARC'] = 'YES'
  s['CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER'] = 'NO'
  s['CLANG_WARN_DOCUMENTATION_COMMENTS'] = 'NO'
  s['IPHONEOS_DEPLOYMENT_TARGET'] = '15.6'
  s['SUPPORTED_PLATFORMS'] = 'iphoneos'
  s['SDKROOT'] = 'iphoneos'
  s['TARGETED_DEVICE_FAMILY'] = '1'
  s['SUPPORTS_MACCATALYST'] = 'NO'
  s['SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD'] = 'NO'
  s['CODE_SIGN_STYLE'] = 'Automatic'
  s['DEVELOPMENT_TEAM'] = '' # Choose your Apple development team in Xcode.
  s['ENABLE_BITCODE'] = 'NO'
  s['ENABLE_USER_SCRIPT_SANDBOXING'] = 'YES'
  s['MARKETING_VERSION'] = '1.0'
  s['CURRENT_PROJECT_VERSION'] = '1'
  s['SWIFT_OPTIMIZATION_LEVEL'] = config.name == 'Release' ? '-O' : '-Onone'
  s['GCC_OPTIMIZATION_LEVEL'] = config.name == 'Release' ? '3' : '0'
  s['SWIFT_ACTIVE_COMPILATION_CONDITIONS'] = 'DEBUG' if config.name == 'Debug'
end
project.save
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(target)
scheme.set_launch_target(target)
scheme.launch_action.build_configuration = 'Release'
scheme.launch_action.xml_element.attributes['selectedDebuggerIdentifier'] = ''
scheme.launch_action.xml_element.attributes['selectedLauncherIdentifier'] = 'Xcode.IDEFoundation.Launcher.PosixSpawn'
scheme.profile_action.build_configuration = 'Release'
scheme.save_as(project.path, 'RuntimeTest', true)
puts 'Generated RuntimeTest.xcodeproj — physical iPhone, Release, debugger disabled.'
