/*
    hololens_fake_test.c

    Single-file Windows C version of the Python motion test.

    It contains only the parts of heq_gimbal.py that the motion test actually
    uses: HEQ header checksum, HEQ CRC32, packet construction, and 0x85 control
    payload construction.

    Current wireless topology:
        PC -> Radio A -> Radio B -> Gimbal

    The reverse telemetry wire is intentionally not used in this program.
*/

#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Change these three values for your tests. */
#define PORT_NAME "\\\\.\\COM4"
#define BAUD_RATE 115200
#define UPDATE_HZ 10.0

#define TEST_DURATION_SECONDS 30.0

#define PITCH_MIN (-135.0)
#define PITCH_MAX (45.0)
#define YAW_MIN   (-135.0)
#define YAW_MAX   (135.0)
#define ROLL_MIN  (-45.0)
#define ROLL_MAX  (45.0)

static HANDLE serial_port = INVALID_HANDLE_VALUE;
static LARGE_INTEGER timer_frequency;

/* ------------------------------------------------------------------------- */
/* Timing                                                                    */
/* ------------------------------------------------------------------------- */

static double now_seconds(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)timer_frequency.QuadPart;
}

static void sleep_until(double deadline)
{
    for (;;)
    {
        double remaining = deadline - now_seconds();

        if (remaining <= 0.0)
            return;

        if (remaining > 0.002)
        {
            DWORD milliseconds = (DWORD)((remaining - 0.001) * 1000.0);

            if (milliseconds < 1)
                milliseconds = 1;

            Sleep(milliseconds);
        }
        else
        {
            SwitchToThread();
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

static double clamp_value(double value, double minimum, double maximum)
{
    if (value < minimum)
        return minimum;

    if (value > maximum)
        return maximum;

    return value;
}

static int16_t to_hundredths(double value)
{
    double scaled = value * 100.0;

    if (scaled > 32767.0)
        scaled = 32767.0;
    else if (scaled < -32768.0)
        scaled = -32768.0;

    return (int16_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

static void write_int16_le(uint8_t *destination, int16_t value)
{
    uint16_t unsigned_value = (uint16_t)value;

    destination[0] = (uint8_t)(unsigned_value & 0xFFU);
    destination[1] = (uint8_t)((unsigned_value >> 8) & 0xFFU);
}

static void write_uint32_le(uint8_t *destination, uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8) & 0xFFU);
    destination[2] = (uint8_t)((value >> 16) & 0xFFU);
    destination[3] = (uint8_t)((value >> 24) & 0xFFU);
}

/* ------------------------------------------------------------------------- */
/* HEQ CRC32                                                                 */
/* ------------------------------------------------------------------------- */

/*
    This generates the same 256-entry table used by heq_gimbal.py.
    Polynomial: 0x04C11DB7
*/
static uint32_t crc32_table[256];

static void initialize_crc32_table(void)
{
    uint32_t index;

    for (index = 0; index < 256; ++index)
    {
        uint32_t crc = index << 24;
        int bit;

        for (bit = 0; bit < 8; ++bit)
        {
            if ((crc & 0x80000000U) != 0)
                crc = (crc << 1) ^ 0x04C11DB7U;
            else
                crc <<= 1;
        }

        crc32_table[index] = crc;
    }
}

static uint32_t calculate_heq_crc32(const uint8_t *data, size_t length)
{
    uint32_t register_value = 0xFFFFFFFFU;
    size_t byte_index;

    for (byte_index = 0; byte_index < length; ++byte_index)
    {
        int repetition;

        register_value ^= data[byte_index];

        /*
            This deliberately repeats four table operations per byte because
            that is exactly what the supplied HEQ Python implementation does.
        */
        for (repetition = 0; repetition < 4; ++repetition)
        {
            uint32_t table_value =
                crc32_table[(register_value >> 24) & 0xFFU];

            register_value = (register_value << 8) ^ table_value;
        }
    }

    return register_value;
}

/* ------------------------------------------------------------------------- */
/* HEQ packet construction                                                   */
/* ------------------------------------------------------------------------- */

static uint8_t calculate_header_checksum(
    uint8_t version,
    uint8_t length,
    uint8_t command
)
{
    return (uint8_t)((version + length + command) & 0xFFU);
}

static size_t build_heq_packet(
    uint8_t command,
    const uint8_t *payload,
    uint8_t payload_length,
    uint8_t *packet,
    size_t packet_capacity
)
{
    const uint8_t frame_header = 0xAE;
    const uint8_t version = 0x01;

    size_t required_length =
        5U + payload_length + (payload_length > 0 ? 4U : 0U);

    size_t position = 0;

    if (packet == NULL || packet_capacity < required_length)
        return 0;

    packet[position++] = frame_header;
    packet[position++] = version;
    packet[position++] = payload_length;
    packet[position++] = command;
    packet[position++] =
        calculate_header_checksum(version, payload_length, command);

    if (payload_length > 0)
    {
        uint32_t crc;

        memcpy(packet + position, payload, payload_length);
        position += payload_length;

        crc = calculate_heq_crc32(payload, payload_length);
        write_uint32_le(packet + position, crc);
        position += 4;
    }

    return position;
}

/*
    0x85 payload:
        byte 0      mode
        bytes 1-2   roll angle
        bytes 3-4   pitch angle
        bytes 5-6   yaw angle
        bytes 7-8   roll speed
        bytes 9-10  pitch speed
        bytes 11-12 yaw speed

    Angles and speeds are signed int16 values in 0.01-degree units.
*/
static void build_0x85_payload(
    int8_t mode,
    double roll_angle_deg,
    double pitch_angle_deg,
    double yaw_angle_deg,
    double roll_speed_deg_s,
    double pitch_speed_deg_s,
    double yaw_speed_deg_s,
    uint8_t payload[13]
)
{
    payload[0] = (uint8_t)mode;

    write_int16_le(payload + 1, to_hundredths(roll_angle_deg));
    write_int16_le(payload + 3, to_hundredths(pitch_angle_deg));
    write_int16_le(payload + 5, to_hundredths(yaw_angle_deg));

    write_int16_le(payload + 7, to_hundredths(roll_speed_deg_s));
    write_int16_le(payload + 9, to_hundredths(pitch_speed_deg_s));
    write_int16_le(payload + 11, to_hundredths(yaw_speed_deg_s));
}

static int send_0x85(
    int8_t mode,
    double roll_angle_deg,
    double pitch_angle_deg,
    double yaw_angle_deg
)
{
    uint8_t payload[13];
    uint8_t packet[22];
    size_t packet_length;
    DWORD bytes_written = 0;

    build_0x85_payload(
        mode,
        roll_angle_deg,
        pitch_angle_deg,
        yaw_angle_deg,
        0.0,
        0.0,
        0.0,
        payload
    );

    packet_length = build_heq_packet(
        0x85,
        payload,
        (uint8_t)sizeof(payload),
        packet,
        sizeof(packet)
    );

    if (packet_length == 0)
        return 0;

    if (!WriteFile(
            serial_port,
            packet,
            (DWORD)packet_length,
            &bytes_written,
            NULL))
    {
        fprintf(stderr, "WriteFile failed. Win32 error %lu\n", GetLastError());
        return 0;
    }

    /*
        Wait until Windows has handed the queued output to the serial driver.
        This mirrors ser.flush() in the Python version.
    */
    if (!FlushFileBuffers(serial_port))
    {
        fprintf(
            stderr,
            "FlushFileBuffers failed. Win32 error %lu\n",
            GetLastError()
        );
        return 0;
    }

    return bytes_written == packet_length;
}

/* ------------------------------------------------------------------------- */
/* Windows serial port                                                       */
/* ------------------------------------------------------------------------- */

static int open_serial_port(void)
{
    DCB settings;
    COMMTIMEOUTS timeouts;

    serial_port = CreateFileA(
        PORT_NAME,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (serial_port == INVALID_HANDLE_VALUE)
    {
        fprintf(
            stderr,
            "Could not open %s. Win32 error %lu\n",
            PORT_NAME,
            GetLastError()
        );
        return 0;
    }

    memset(&settings, 0, sizeof(settings));
    settings.DCBlength = sizeof(settings);

    if (!GetCommState(serial_port, &settings))
    {
        fprintf(
            stderr,
            "GetCommState failed. Win32 error %lu\n",
            GetLastError()
        );
        CloseHandle(serial_port);
        serial_port = INVALID_HANDLE_VALUE;
        return 0;
    }

    settings.BaudRate = BAUD_RATE;
    settings.ByteSize = 8;
    settings.Parity = NOPARITY;
    settings.StopBits = ONESTOPBIT;

    settings.fBinary = TRUE;
    settings.fParity = FALSE;

    settings.fOutxCtsFlow = FALSE;
    settings.fOutxDsrFlow = FALSE;
    settings.fDtrControl = DTR_CONTROL_DISABLE;
    settings.fDsrSensitivity = FALSE;

    settings.fOutX = FALSE;
    settings.fInX = FALSE;
    settings.fRtsControl = RTS_CONTROL_DISABLE;

    if (!SetCommState(serial_port, &settings))
    {
        fprintf(
            stderr,
            "SetCommState failed. Win32 error %lu\n",
            GetLastError()
        );
        CloseHandle(serial_port);
        serial_port = INVALID_HANDLE_VALUE;
        return 0;
    }

    memset(&timeouts, 0, sizeof(timeouts));

    timeouts.ReadIntervalTimeout = 30;
    timeouts.ReadTotalTimeoutConstant = 30;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = 100;
    timeouts.WriteTotalTimeoutMultiplier = 0;

    if (!SetCommTimeouts(serial_port, &timeouts))
    {
        fprintf(
            stderr,
            "SetCommTimeouts failed. Win32 error %lu\n",
            GetLastError()
        );
        CloseHandle(serial_port);
        serial_port = INVALID_HANDLE_VALUE;
        return 0;
    }

    PurgeComm(serial_port, PURGE_RXCLEAR | PURGE_TXCLEAR);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Gimbal motion test                                                        */
/* ------------------------------------------------------------------------- */

static void return_to_center(double wait_seconds)
{
    printf("Returning to center...\n");

    if (!send_0x85(3, 0.0, 0.0, 0.0))
        fprintf(stderr, "Failed to send return-to-center command.\n");

    Sleep((DWORD)(wait_seconds * 1000.0));
}

static void run_dynamic_profile(double duration_seconds)
{
    const double interval = 1.0 / UPDATE_HZ;

    double start = now_seconds();
    double next_tick = start;
    double last_print = 0.0;

    printf(
        "Running dynamic HoloLens-style profile for %.1f s at %.1f Hz\n",
        duration_seconds,
        UPDATE_HZ
    );

    for (;;)
    {
        double now = now_seconds();
        double elapsed = now - start;

        double yaw;
        double pitch;
        double roll;
        double phase;
        double sleep_deadline;

        if (elapsed >= duration_seconds)
            break;

        yaw =
            85.0 * sin(2.0 * M_PI * 0.095 * elapsed) +
            28.0 * sin(2.0 * M_PI * 0.29 * elapsed + 0.8) +
            10.0 * sin(2.0 * M_PI * 0.75 * elapsed + 1.9);

        pitch =
            -35.0 +
            45.0 * sin(2.0 * M_PI * 0.17 * elapsed + 0.5) +
            20.0 * sin(2.0 * M_PI * 0.43 * elapsed + 2.1);

        roll =
            28.0 * sin(2.0 * M_PI * 0.21 * elapsed + 1.2) +
            12.0 * sin(2.0 * M_PI * 0.62 * elapsed + 0.1) +
             5.0 * sin(2.0 * M_PI * 1.10 * elapsed + 2.0);

        phase = fmod(elapsed, 10.0);

        if (phase > 1.5 && phase < 2.6)
        {
            yaw += 30.0;
            pitch += 20.0;
            roll += 18.0;
        }
        else if (phase > 4.2 && phase < 5.4)
        {
            yaw -= 40.0;
            pitch -= 35.0;
            roll -= 20.0;
        }
        else if (phase > 7.0 && phase < 8.1)
        {
            yaw += 18.0;
            pitch -= 10.0;
            roll += 24.0;
        }

        roll = clamp_value(roll, ROLL_MIN, ROLL_MAX);
        pitch = clamp_value(pitch, PITCH_MIN, PITCH_MAX);
        yaw = clamp_value(yaw, YAW_MIN, YAW_MAX);

        if (!send_0x85(2, roll, pitch, yaw))
            fprintf(stderr, "Failed to send angle command.\n");

        if (now - last_print > 0.25)
        {
            printf(
                "[CMD] target r/p/y = %.2f, %.2f, %.2f deg\n",
                roll,
                pitch,
                yaw
            );

            last_print = now;
        }

        next_tick += interval;
        sleep_deadline = next_tick;

        if (sleep_deadline > now_seconds())
            sleep_until(sleep_deadline);
    }

    printf("Profile complete.\n");
}

int main(void)
{
    QueryPerformanceFrequency(&timer_frequency);
    initialize_crc32_table();

    if (!open_serial_port())
        return 1;

    printf("Opened COM4 at 115200 baud.\n");
    printf("Command rate: %.1f Hz\n", UPDATE_HZ);
    printf("Telemetry is not read by this one-way wireless test.\n");

    Sleep(1000);

    return_to_center(2.0);
    run_dynamic_profile(TEST_DURATION_SECONDS);
    return_to_center(3.0);

    CloseHandle(serial_port);
    return 0;
}
