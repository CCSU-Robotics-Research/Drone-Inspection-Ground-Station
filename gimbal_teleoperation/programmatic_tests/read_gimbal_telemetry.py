import sys
import serial
from heq_gimbal import HEQParser, decode_0x87_v2

PORT = "COM4"
BAUD = 115200

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0.1,
)

parser = HEQParser()

def main():
    if len(sys.argv) > 1:
        outfile = sys.argv[1]
        sys.stdout = open(outfile, "w")

    print(f"Listening on {PORT}...")

    try:
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue

            frames = parser.feed(chunk)
            for frame in frames:
                status = f"hdr_ok={frame.header_ok} crc_ok={frame.crc_ok}"
                print(f"CMD=0x{frame.command:02X} LEN={frame.length} {status}")
                print(f"  RAW: {frame.raw.hex(' ').upper()}")

                if frame.command == 0x87 and frame.length == 24 and frame.header_ok and frame.crc_ok:
                    telem = decode_0x87_v2(frame.data)
                    print(
                        f"  IMU r/p/y = "
                        f"{telem['imu_roll']:.2f}, "
                        f"{telem['imu_pitch']:.2f}, "
                        f"{telem['imu_yaw']:.2f} deg"
                    )
                    print(
                        f"  Hall r/p/y = "
                        f"{telem['hall_roll']:.2f}, "
                        f"{telem['hall_pitch']:.2f}, "
                        f"{telem['hall_yaw']:.2f} deg"
                    )
                else:
                    print("  This frame was unable to be decoded.")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        ser.close()


if __name__ == "__main__":
    main()