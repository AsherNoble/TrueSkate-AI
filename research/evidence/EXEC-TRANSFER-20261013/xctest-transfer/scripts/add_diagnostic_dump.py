from pathlib import Path
import difflib
ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'resume-20261013/completed.json').exists(),'Finish unmodified WDA measurement first'
p=ROOT/'wda/WebDriverAgentLib/Utilities/FBW3CActionsSynthesizer.m'
a=p.read_text(); (ROOT/'resume-20261013/FBW3CActionsSynthesizer-original.m').write_text(a)
assert a.count('  return eventRecord;')==1
code='''  // Isolated L0 simulator instrumentation: inspect the record without mutating it.
  NSMutableArray *l0Paths = [NSMutableArray array];
  for (XCPointerEventPath *l0Path in eventRecord.eventPaths) {
    NSMutableArray *l0Events = [NSMutableArray array];
    for (id l0Event in l0Path.pointerEvents) {
      CGPoint l0Point = [[l0Event valueForKey:@"coordinate"] CGPointValue];
      [l0Events addObject:@{@"offset_s": [l0Event valueForKey:@"offset"],
                            @"event_type": [l0Event valueForKey:@"eventType"],
                            @"x": @(l0Point.x), @"y": @(l0Point.y)}];
    }
    [l0Paths addObject:l0Events];
  }
  static NSUInteger l0Sequence = 0;
  NSDictionary *l0Dump = @{@"sequence": @(l0Sequence++),
                            @"maximum_offset_s": @(eventRecord.maximumOffset),
                            @"paths": l0Paths};
  NSData *l0Data = [NSJSONSerialization dataWithJSONObject:l0Dump options:0 error:nil];
  NSLog(@"L0_SYNTH_RECORD %@", [[NSString alloc] initWithData:l0Data encoding:NSUTF8StringEncoding]);
  return eventRecord;'''
b=a.replace('  return eventRecord;',code)
p.write_text(b);(ROOT/'diagnostic-dump.patch').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='ae50404a/FBW3CActionsSynthesizer.m',tofile='L0-diagnostic/FBW3CActionsSynthesizer.m')))
print('Patched isolated copied WDA only')
