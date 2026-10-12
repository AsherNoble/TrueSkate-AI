// Physical UIKit diagnostic. No filesystem IO or formatted output during movement.
// CADisplayLink is a scheduled callback, not proof of panel/photon presentation.
#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>
#include <stdio.h>
#ifndef LOGGER_SOURCE_SHA256
#define LOGGER_SOURCE_SHA256 "unbound"
#endif

typedef struct {
  uint64_t sequence, callback, touch;
  char kind, phase;
  double now, uptime, timestamp, target, x, y;
} Sample;
static Sample samples[16384];
static size_t count;
static uint64_t sequence, callbackSequence, batch;
static BOOL overflow;
static dispatch_queue_t writer;
static FILE *output;

static char phaseName(UITouchPhase p) {
  switch (p) {
    case UITouchPhaseBegan: return 'B';
    case UITouchPhaseMoved: return 'M';
    case UITouchPhaseStationary: return 'S';
    case UITouchPhaseEnded: return 'E';
    case UITouchPhaseCancelled: return 'C';
    default: return '?';
  }
}

static void append(char kind, char phase, uint64_t cb, uint64_t touch,
                   double now, double ts, double target, CGPoint p) {
  if (count == sizeof(samples)/sizeof(samples[0])) { overflow = YES; return; }
  samples[count++] = (Sample){++sequence, cb, touch, kind, phase, now,
    NSProcessInfo.processInfo.systemUptime, ts, target, p.x, p.y};
}

static void flushBatch(const char *reason) {
  if (!count && !overflow) return;
  // Copy only after release or cancellation; JSON serialization and writes are off-main.
  NSData *snapshot = [NSData dataWithBytes:samples length:count*sizeof(Sample)];
  BOOL lost = overflow;
  uint64_t number = ++batch;
  NSString *why = [NSString stringWithUTF8String:reason];
  count = 0; overflow = NO;
  dispatch_async(writer, ^{
    fprintf(output, "{\"kind\":\"batch\",\"batch\":%llu,\"reason\":\"%s\",\"overflow\":%s}\n",
            number, why.UTF8String, lost ? "true" : "false");
    const Sample *p = snapshot.bytes;
    for (size_t i=0; i<snapshot.length/sizeof(Sample); ++i) {
      const Sample s=p[i];
      fprintf(output, "{\"kind\":\"%c\",\"phase\":\"%c\",\"seq\":%llu,\"cb\":%llu,\"tid\":%llu,"
          "\"now\":%.9f,\"uptime\":%.9f,\"ts\":%.9f,\"target\":%.9f,\"x\":%.6f,\"y\":%.6f}\n",
          s.kind,s.phase,s.sequence,s.callback,s.touch,s.now,s.uptime,s.timestamp,s.target,s.x,s.y);
    }
    fflush(output);
  });
}

@interface LogView : UIView
@property BOOL contactActive;
@property CGPoint displayedPoint;
@property CADisplayLink *displayLink;
@end
@implementation LogView
- (instancetype)initWithFrame:(CGRect)frame {
  if ((self=[super initWithFrame:frame])) {
    self.backgroundColor=UIColor.whiteColor;
    self.multipleTouchEnabled=NO;
    self.displayLink=[CADisplayLink displayLinkWithTarget:self selector:@selector(tick:)];
    [self.displayLink addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
  }
  return self;
}
- (void)tick:(CADisplayLink *)link {
  if (self.contactActive)
    append('D','-',0,0,CACurrentMediaTime(),link.timestamp,link.targetTimestamp,self.displayedPoint);
}
- (void)logTouches:(NSSet<UITouch *> *)touches event:(UIEvent *)event {
  const double now=CACurrentMediaTime();
  const uint64_t cb=++callbackSequence;
  for (UITouch *t in touches) {
    CGPoint p=[t preciseLocationInView:self];
    if (t.phase==UITouchPhaseBegan) self.contactActive=YES;
    append('T',phaseName(t.phase),cb,t.hash,now,t.timestamp,0,p);
    for (UITouch *c in [event coalescedTouchesForTouch:t])
      append('C',phaseName(c.phase),cb,t.hash,now,c.timestamp,0,[c preciseLocationInView:self]);
    self.displayedPoint=p;
    if (t.phase==UITouchPhaseEnded || t.phase==UITouchPhaseCancelled) {
      self.contactActive=NO;
      flushBatch(t.phase==UITouchPhaseEnded ? "release" : "cancelled");
    }
  }
  [self setNeedsDisplay];
}
- (void)touchesBegan:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e]; }
- (void)touchesMoved:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e]; }
- (void)touchesEnded:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e]; }
- (void)touchesCancelled:(NSSet<UITouch *> *)t withEvent:(UIEvent *)e { [self logTouches:t event:e]; }
- (void)drawRect:(CGRect)rect {
  if (self.contactActive) {
    [UIColor.orangeColor setFill];
    UIBezierPath *dot=[UIBezierPath bezierPathWithOvalInRect:CGRectMake(
      self.displayedPoint.x-5,self.displayedPoint.y-5,10,10)];
    [dot fill];
  }
}
@end

@interface VC : UIViewController @end
@implementation VC
- (void)loadView { self.view=[[LogView alloc] initWithFrame:UIScreen.mainScreen.bounds]; }
- (BOOL)prefersStatusBarHidden { return YES; }
- (BOOL)prefersHomeIndicatorAutoHidden { return YES; }
- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures { return UIRectEdgeAll; }
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property(strong,nonatomic) UIWindow *window;
@end
@implementation AppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
  NSString *docs=NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask,YES).firstObject;
  NSString *filename=[NSString stringWithFormat:@"touches-%@.jsonl",NSUUID.UUID.UUIDString];
  output=fopen([docs stringByAppendingPathComponent:filename].fileSystemRepresentation,"wx");
  if (!output) return NO;
  writer=dispatch_queue_create("physical-transfer-writer",DISPATCH_QUEUE_SERIAL);
  CGRect b=UIScreen.mainScreen.bounds;
  fprintf(output,"{\"kind\":\"launch\",\"schema\":\"physical-touch-v1\",\"now\":%.9f,\"uptime\":%.9f,"
    "\"w\":%.1f,\"h\":%.1f,\"scale\":%.1f,\"maximum_fps\":%ld,\"capacity\":16384,"
    "\"epoch\":%.9f,\"source_sha256\":\"%s\",\"ios_version\":\"%s\"}\n",
    CACurrentMediaTime(),NSProcessInfo.processInfo.systemUptime,b.size.width,b.size.height,
    UIScreen.mainScreen.scale,(long)UIScreen.mainScreen.maximumFramesPerSecond,
    NSDate.date.timeIntervalSince1970,LOGGER_SOURCE_SHA256,UIDevice.currentDevice.systemVersion.UTF8String);
  fflush(output);
  self.window=[[UIWindow alloc] initWithFrame:b];
  self.window.rootViewController=[VC new];
  [self.window makeKeyAndVisible];
  return YES;
}
- (void)applicationDidEnterBackground:(UIApplication *)app { flushBatch("background"); }
@end

int main(int argc,char *argv[]) {
  @autoreleasepool { return UIApplicationMain(argc,argv,nil,NSStringFromClass(AppDelegate.class)); }
}
