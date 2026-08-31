package com.touch;

import android.os.IBinder;
import android.os.SystemClock;
import android.view.InputDevice;
import android.view.InputEvent;
import android.view.MotionEvent;
import java.lang.reflect.Method;

public class PinchZoom {
    public static void main(String[] args) {
        try {
            int screenW = args.length > 0 ? Integer.parseInt(args[0]) : 1264;
            int screenH = args.length > 1 ? Integer.parseInt(args[1]) : 2736;
            int durationMs = args.length > 2 ? Integer.parseInt(args[2]) : 350;
            int steps = 25;

            // Get IInputManager via ServiceManager (universal across all Android versions 8 - 15)
            Class<?> smClass = Class.forName("android.os.ServiceManager");
            Method getService = smClass.getMethod("getService", String.class);
            IBinder binder = (IBinder) getService.invoke(null, "input");

            Class<?> iimClass = Class.forName("android.hardware.input.IInputManager$Stub");
            Method asInterface = iimClass.getMethod("asInterface", IBinder.class);
            Object im = asInterface.invoke(null, binder);

            // Find injectInputEvent method on IInputManager
            Method inject = null;
            for (Method m : im.getClass().getMethods()) {
                if (m.getName().equals("injectInputEvent") && m.getParameterTypes().length >= 2) {
                    inject = m;
                    break;
                }
            }

            int cy = (int)(screenH * 0.575);
            float f1StartX = screenW * 0.12f;
            float f1EndX = screenW * 0.40f;
            float f2StartX = screenW * 0.88f;
            float f2EndX = screenW * 0.60f;

            long downTime = SystemClock.uptimeMillis();

            MotionEvent.PointerProperties[] props = new MotionEvent.PointerProperties[2];
            props[0] = new MotionEvent.PointerProperties();
            props[0].id = 0;
            props[0].toolType = MotionEvent.TOOL_TYPE_FINGER;

            props[1] = new MotionEvent.PointerProperties();
            props[1].id = 1;
            props[1].toolType = MotionEvent.TOOL_TYPE_FINGER;

            MotionEvent.PointerCoords[] coords = new MotionEvent.PointerCoords[2];
            coords[0] = new MotionEvent.PointerCoords();
            coords[0].x = f1StartX;
            coords[0].y = cy;
            coords[0].pressure = 1.0f;
            coords[0].size = 1.0f;

            coords[1] = new MotionEvent.PointerCoords();
            coords[1].x = f2StartX;
            coords[1].y = cy;
            coords[1].pressure = 1.0f;
            coords[1].size = 1.0f;

            // Helper to invoke injectInputEvent
            Object[] injectArgs;
            if (inject.getParameterTypes().length == 2) {
                injectArgs = new Object[2];
                injectArgs[1] = 0; // INJECT_INPUT_EVENT_MODE_ASYNC
            } else {
                injectArgs = new Object[inject.getParameterTypes().length];
                injectArgs[1] = 0;
            }

            // 1. Pointer 0 DOWN
            MotionEvent evDown0 = MotionEvent.obtain(
                downTime, downTime, MotionEvent.ACTION_DOWN, 1,
                props, coords, 0, 0, 1.0f, 1.0f, 0, 0,
                InputDevice.SOURCE_TOUCHSCREEN, 0
            );
            injectArgs[0] = evDown0;
            inject.invoke(im, injectArgs);

            // 2. Pointer 1 DOWN (ACTION_POINTER_DOWN)
            long t = SystemClock.uptimeMillis();
            int actionPointerDown = MotionEvent.ACTION_POINTER_DOWN | (1 << MotionEvent.ACTION_POINTER_INDEX_SHIFT);
            MotionEvent evDown1 = MotionEvent.obtain(
                downTime, t, actionPointerDown, 2,
                props, coords, 0, 0, 1.0f, 1.0f, 0, 0,
                InputDevice.SOURCE_TOUCHSCREEN, 0
            );
            injectArgs[0] = evDown1;
            inject.invoke(im, injectArgs);

            // 3. Synchronous smooth MOVE steps
            long stepSleep = durationMs / steps;
            for (int i = 1; i <= steps; i++) {
                float alpha = (float) i / steps;
                coords[0].x = f1StartX + (f1EndX - f1StartX) * alpha;
                coords[1].x = f2StartX + (f2EndX - f2StartX) * alpha;
                t = SystemClock.uptimeMillis();

                MotionEvent evMove = MotionEvent.obtain(
                    downTime, t, MotionEvent.ACTION_MOVE, 2,
                    props, coords, 0, 0, 1.0f, 1.0f, 0, 0,
                    InputDevice.SOURCE_TOUCHSCREEN, 0
                );
                injectArgs[0] = evMove;
                inject.invoke(im, injectArgs);

                if (stepSleep > 0) {
                    Thread.sleep(stepSleep);
                }
            }

            // 4. Pointer 1 UP (ACTION_POINTER_UP)
            t = SystemClock.uptimeMillis();
            int actionPointerUp = MotionEvent.ACTION_POINTER_UP | (1 << MotionEvent.ACTION_POINTER_INDEX_SHIFT);
            MotionEvent evUp1 = MotionEvent.obtain(
                downTime, t, actionPointerUp, 2,
                props, coords, 0, 0, 1.0f, 1.0f, 0, 0,
                InputDevice.SOURCE_TOUCHSCREEN, 0
            );
            injectArgs[0] = evUp1;
            inject.invoke(im, injectArgs);

            // 5. Pointer 0 UP (ACTION_UP)
            t = SystemClock.uptimeMillis();
            MotionEvent evUp0 = MotionEvent.obtain(
                downTime, t, MotionEvent.ACTION_UP, 1,
                props, coords, 0, 0, 1.0f, 1.0f, 0, 0,
                InputDevice.SOURCE_TOUCHSCREEN, 0
            );
            injectArgs[0] = evUp0;
            inject.invoke(im, injectArgs);

            System.out.println("[+] Synchronized multi-touch pinch executed successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
