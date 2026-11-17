/*
 * xtc2bin.c -- stand-alone GROMACS XTC decoder (no external libraries).
 *
 * Port of the xdr3dfcoord decompression algorithm of the public-domain
 * xdrfile library (GROMACS).  Writes
 *     <out>.coords.f32 : float32 coordinates in Angstrom, frame-major
 *                        (nframes x natoms x 3)
 *     <out>.frames.txt : frame index, step, time (ps), box (9 values, Angstrom)
 * Usage:  xtc2bin input.xtc out_prefix [stride]
 * Build:  gcc -O2 -o xtc2bin xtc2bin.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const int magicints[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216};
#define FIRSTIDX 9
#define LASTIDX ((int)(sizeof(magicints) / sizeof(*magicints)))

static int read_be_int(FILE *f, int *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (int)((uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | (uint32_t)b[3]);
    return 1;
}
static int read_be_float(FILE *f, float *v) {
    int i; if (!read_be_int(f, &i)) return 0;
    memcpy(v, &i, 4); return 1;
}

/* bit-reader state */
typedef struct { const unsigned char *buf; int cnt; unsigned int lastbits, lastbyte; } bitstate;

static int receivebits(bitstate *s, int num_of_bits) {
    int num = 0, mask = (num_of_bits >= 32) ? -1 : ((1 << num_of_bits) - 1);
    while (num_of_bits >= 8) {
        s->lastbyte = (s->lastbyte << 8) | s->buf[s->cnt++];
        num |= (s->lastbyte >> s->lastbits) << (num_of_bits - 8);
        num_of_bits -= 8;
    }
    if (num_of_bits > 0) {
        if ((int)s->lastbits < num_of_bits) {
            s->lastbits += 8;
            s->lastbyte = (s->lastbyte << 8) | s->buf[s->cnt++];
        }
        s->lastbits -= num_of_bits;
        num |= (s->lastbyte >> s->lastbits) & ((1 << num_of_bits) - 1);
    }
    return num & mask;
}

static void receiveints(bitstate *s, int num_of_ints, int num_of_bits,
                        const unsigned int sizes[], int nums[]) {
    int bytes[32], i, j, num_of_bytes = 0, p, num;
    bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
    while (num_of_bits > 8) { bytes[num_of_bytes++] = receivebits(s, 8); num_of_bits -= 8; }
    if (num_of_bits > 0) bytes[num_of_bytes++] = receivebits(s, num_of_bits);
    for (i = num_of_ints - 1; i > 0; i--) {
        num = 0;
        for (j = num_of_bytes - 1; j >= 0; j--) {
            num = (num << 8) | bytes[j];
            p = num / sizes[i];
            bytes[j] = p;
            num = num - p * sizes[i];
        }
        nums[i] = num;
    }
    nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

static int sizeofint(int size) {
    unsigned int num = 1; int nb = 0;
    while (size >= (int)num && nb < 32) { nb++; num <<= 1; }
    return nb;
}
static int sizeofints(int num_of_ints, const unsigned int sizes[]) {
    int i, num; unsigned int num_of_bytes = 1, num_of_bits = 0, bytes[32], bytecnt, tmp;
    bytes[0] = 1;
    for (i = 0; i < num_of_ints; i++) {
        tmp = 0;
        for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff; tmp >>= 8;
        }
        while (tmp != 0) { bytes[bytecnt++] = tmp & 0xff; tmp >>= 8; }
        num_of_bytes = bytecnt;
    }
    num = 1; num_of_bytes--;
    while ((int)bytes[num_of_bytes] >= num) { num_of_bits++; num *= 2; }
    return num_of_bits + num_of_bytes * 8;
}

static int decompress(FILE *f, int natoms, float *out) {
    float precision; int minint[3], maxint[3], smallidx, byte_cnt, i, k;
    unsigned int sizeint[3], sizesmall[3], bitsizeint[3] = {0, 0, 0};
    int bitsize;
    if (!read_be_float(f, &precision)) return 0;
    for (i = 0; i < 3; i++) if (!read_be_int(f, &minint[i])) return 0;
    for (i = 0; i < 3; i++) if (!read_be_int(f, &maxint[i])) return 0;
    for (i = 0; i < 3; i++) sizeint[i] = maxint[i] - minint[i] + 1;
    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
        for (i = 0; i < 3; i++) bitsizeint[i] = sizeofint(sizeint[i]);
        bitsize = 0;
    } else bitsize = sizeofints(3, sizeint);
    if (!read_be_int(f, &smallidx)) return 0;
    int tmpidx = smallidx - 1; if (tmpidx < FIRSTIDX) tmpidx = FIRSTIDX;
    int smaller = magicints[tmpidx] / 2;
    int smallnum = magicints[smallidx] / 2;
    sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];
    if (!read_be_int(f, &byte_cnt)) return 0;
    int padded = (byte_cnt + 3) & ~3;
    unsigned char *buf = (unsigned char *)malloc(padded + 8);
    if (fread(buf, 1, padded, f) != (size_t)padded) { free(buf); return 0; }
    bitstate s = {buf, 0, 0, 0};
    float inv = 1.0f / precision;
    int run = 0, thiscoord[3], prevcoord[3], tmp;
    float *lfp = out;
    i = 0;
    while (i < natoms) {
        if (bitsize == 0) {
            thiscoord[0] = receivebits(&s, bitsizeint[0]);
            thiscoord[1] = receivebits(&s, bitsizeint[1]);
            thiscoord[2] = receivebits(&s, bitsizeint[2]);
        } else receiveints(&s, 3, bitsize, sizeint, thiscoord);
        i++;
        thiscoord[0] += minint[0]; thiscoord[1] += minint[1]; thiscoord[2] += minint[2];
        prevcoord[0] = thiscoord[0]; prevcoord[1] = thiscoord[1]; prevcoord[2] = thiscoord[2];
        int flag = receivebits(&s, 1), is_smaller = 0;
        if (flag == 1) {
            run = receivebits(&s, 5);
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller--;
        }
        if (run > 0) {
            for (k = 0; k < run; k += 3) {
                receiveints(&s, 3, smallidx, sizesmall, thiscoord);
                i++;
                thiscoord[0] += prevcoord[0] - smallnum;
                thiscoord[1] += prevcoord[1] - smallnum;
                thiscoord[2] += prevcoord[2] - smallnum;
                if (k == 0) {
                    tmp = thiscoord[0]; thiscoord[0] = prevcoord[0]; prevcoord[0] = tmp;
                    tmp = thiscoord[1]; thiscoord[1] = prevcoord[1]; prevcoord[1] = tmp;
                    tmp = thiscoord[2]; thiscoord[2] = prevcoord[2]; prevcoord[2] = tmp;
                    *lfp++ = prevcoord[0] * inv; *lfp++ = prevcoord[1] * inv; *lfp++ = prevcoord[2] * inv;
                } else {
                    prevcoord[0] = thiscoord[0]; prevcoord[1] = thiscoord[1]; prevcoord[2] = thiscoord[2];
                }
                *lfp++ = thiscoord[0] * inv; *lfp++ = thiscoord[1] * inv; *lfp++ = thiscoord[2] * inv;
            }
        } else {
            *lfp++ = thiscoord[0] * inv; *lfp++ = thiscoord[1] * inv; *lfp++ = thiscoord[2] * inv;
        }
        smallidx += is_smaller;
        if (is_smaller < 0) {
            smallnum = smaller;
            smaller = (smallidx > FIRSTIDX) ? magicints[smallidx - 1] / 2 : 0;
        } else if (is_smaller > 0) {
            smaller = smallnum;
            smallnum = magicints[smallidx] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];
    }
    free(buf);
    return (int)((lfp - out) / 3) == natoms;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.xtc outprefix [stride]\n", argv[0]); return 1; }
    int stride = argc > 3 ? atoi(argv[3]) : 1;
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    char fn[1024];
    snprintf(fn, sizeof fn, "%s.coords.f32", argv[2]); FILE *fc = fopen(fn, "wb");
    snprintf(fn, sizeof fn, "%s.frames.txt", argv[2]); FILE *ft = fopen(fn, "w");
    fprintf(ft, "# frame step time_ps box_xx xy xz yx yy yz zx zy zz (Angstrom)\n");
    int magic, natoms, step, lsize, frame = 0, written = 0; float time, box[9];
    float *x = NULL;
    while (read_be_int(f, &magic)) {
        if (magic != 1995) { fprintf(stderr, "bad magic %d at frame %d\n", magic, frame); return 2; }
        read_be_int(f, &natoms); read_be_int(f, &step); read_be_float(f, &time);
        for (int i = 0; i < 9; i++) read_be_float(f, &box[i]);
        read_be_int(f, &lsize);
        if (!x) x = (float *)malloc(sizeof(float) * 3 * natoms);
        if (natoms <= 9) {
            for (int i = 0; i < 3 * natoms; i++) read_be_float(f, &x[i]);
        } else if (!decompress(f, natoms, x)) { fprintf(stderr, "decompression failed at frame %d\n", frame); return 3; }
        if (frame % stride == 0) {
            for (int i = 0; i < 3 * natoms; i++) x[i] *= 10.0f;      /* nm -> Angstrom */
            fwrite(x, sizeof(float), 3 * natoms, fc);
            fprintf(ft, "%d %d %.4f", frame, step, time);
            for (int i = 0; i < 9; i++) fprintf(ft, " %.4f", box[i] * 10.0f);
            fprintf(ft, "\n"); written++;
        }
        frame++;
    }
    fprintf(stderr, "natoms %d, frames read %d, written %d\n", natoms, frame, written);
    fclose(f); fclose(fc); fclose(ft); free(x);
    return 0;
}
