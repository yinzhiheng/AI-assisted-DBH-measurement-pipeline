#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
NS_ASSUME_NONNULL_BEGIN
@interface RuntimeBridge : NSObject
- (nullable NSString *)loadModels;
- (NSDictionary<NSString *, id> *)runRGB:(NSString *)rgbPath
                                 depth:(NSString *)depthPath
                                 baseX:(NSInteger)baseX
                                 baseY:(NSInteger)baseY
                               preview:(BOOL)preview;
@end
NS_ASSUME_NONNULL_END
