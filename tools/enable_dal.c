/*
 * Injected shim that lets ffmpeg's avfoundation input see iOS devices.
 *
 * Plain ffmpeg cannot enumerate iPhones as capture devices because it never
 * sets the CoreMediaIO kCMIOHardwarePropertyAllowScreenCaptureDevices flag
 * (QuickTime sets it; ffmpeg does not). This constructor sets it at load time.
 *
 * Build:
 *   clang -x c -dynamiclib enable_dal.c -framework CoreMediaIO -o libenable_dal.dylib
 * Use:
 *   DYLD_INSERT_LIBRARIES=/path/libenable_dal.dylib ffmpeg -f avfoundation ...
 */
#include <CoreMediaIO/CMIOHardware.h>
#include <CoreFoundation/CoreFoundation.h>

__attribute__((constructor)) static void enable_dal(void) {
    CMIOObjectPropertyAddress p = {kCMIOHardwarePropertyAllowScreenCaptureDevices,
        kCMIOObjectPropertyScopeGlobal, kCMIOObjectPropertyElementMain};
    UInt32 yes = 1;
    CMIOObjectSetPropertyData(kCMIOObjectSystemObject, &p, 0, NULL, sizeof(yes), &yes);
    /* iOS devices publish asynchronously after the flag flips; ffmpeg
     * enumerates immediately on startup, so pump the run loop briefly to
     * let the DAL plugin register the device in this process. */
    for (int i = 0; i < 30; i++)
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, false);
}
