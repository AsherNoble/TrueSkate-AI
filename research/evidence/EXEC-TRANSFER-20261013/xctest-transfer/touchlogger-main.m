// Minimal touch logger for L0: logs every UITouch callback and its coalesced samples.
#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>
#include <stdio.h>

static FILE *gLog;

static const char *phaseName(UITouchPhase p) {
  switch (p) {
    case UITouchPhaseBegan: return "B";
    case UITouchPhaseMoved: return "M";
    case UITouchPhaseStationary: return "S";
    case UITouchPhaseEnded: return "E";
    case UITouchPhaseCancelled: return "C";
    default: return "?";
  }
}

@interface LogView : UIView
@end

@implementation LogView
- (void)logTouches:(NSSet<UITouch *> *)touches event:(UIEvent *)event cb:(const char *)cb {
  double now = CACurrentMediaTime();
  for (UITouch *t in touches) {
    CGPoint p = [t locationInView:self];
    CGPoint pp = [t preciseLocationInView:self];
    fprintf(gLog, "{\"cb\":\"%s\",\"now\":%.6f,\"ts\":%.6f,\"ph\":\"%s\",\"x\":%.4f,\"y\":%.4f,\"px\":%.4f,\"py\":%.4f,\"tid\":%lu,\"co\":[",
            cb, now, t.timestamp, phaseName(t.phase), p.x, p.y, pp.x, pp.y, (unsigned long)t.hash);
    NSArray<UITouch *> *co = [event coalescedTouchesForTouch:t];
    for (NSUInteger i = 0; i < co.count; i++) {
      UITouch *c = co[i];
      CGPoint cp = [c preciseLocationInView:self];
      fprintf(gLog, "%s[%.6f,%.4f,%.4f,\"%s\"]", i ? "," : "", c.timestamp, cp.x, cp.y, phaseName(c.phase));
    }
    fprintf(gLog, "]}\n");
  }
  fflush(gLog);
}
- (void)touchesBegan:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e cb:"began"]; }
- (void)touchesMoved:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e cb:"moved"]; }
- (void)touchesEnded:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e cb:"ended"]; }
- (void)touchesCancelled:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e cb:"cancelled"]; }
@end

@interface VC : UIViewController
@end
@implementation VC
- (void)loadView { LogView *v = [LogView new]; v.backgroundColor = UIColor.whiteColor; v.multipleTouchEnabled = YES; self.view = v; }
- (BOOL)prefersStatusBarHidden { return YES; }
- (BOOL)prefersHomeIndicatorAutoHidden { return YES; }
- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures { return UIRectEdgeAll; }
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (strong, nonatomic) UIWindow *window;
@end
@implementation AppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
  NSString *docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
  NSString *path = [docs stringByAppendingPathComponent:@"touches.jsonl"];
  gLog = fopen(path.fileSystemRepresentation, "a");
  CGRect b = UIScreen.mainScreen.bounds;
  fprintf(gLog, "{\"cb\":\"launch\",\"now\":%.6f,\"w\":%.1f,\"h\":%.1f,\"scale\":%.1f,\"fps\":%ld}\n",
          CACurrentMediaTime(), b.size.width, b.size.height, UIScreen.mainScreen.scale, (long)UIScreen.mainScreen.maximumFramesPerSecond);
  fflush(gLog);
  self.window = [[UIWindow alloc] initWithFrame:b];
  self.window.rootViewController = [VC new];
  [self.window makeKeyAndVisible];
  return YES;
}
@end

int main(int argc, char *argv[]) {
  @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(AppDelegate.class)); }
}
