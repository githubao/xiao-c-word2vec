// Created by BaoQiang at 2016/11/9 19:20

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define max_size 2000 // string length
#define N 1 // number of closest words
#define max_w 50 // max_length of vocabulary entries

int main4(int argc, char **argv) {
    FILE *f;
    char st1[max_size], st2[max_size], st3[max_size], st4[max_size],
            bestw[N][max_size], file_name[max_size];
    float dist, len, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
    float *M;
    char *vocab;
    int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
    if (argc < 2) {
        printf("Usage: ./compute-accuracy <FILE> <threshold>\n");
        printf("where FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0=off, otherwise typical value is 30000)\n");
        return 0;
    }
    strcpy(file_name, argv[1]);
    if (argc > 2) {
        threshold = atoi(argv[2]);
    }
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) {
        if (words > threshold) {
            words = threshold;
        }
    }
    fscanf(f, "%lld", &size);
    vocab = (char *) malloc(words * max_w * sizeof(char));
    M = (char *) malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) {
                break;
            }
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) {
                a++;
            }
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < max_w; a++) {
            vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        }
        for (a = 0; a < size; a++) {
            fread(&M[b * size + a], sizeof(float), 1, f);
        }
        len = 0;
        for (a = 0; a < size; a++) {
            len += M[b * size + a] * M[b * size + a];
        }
        len = sqrt(len);
        for (a = 0; a < size; a++) {
            M[b * size + a] /= len;
        }
    }
    fclose(f);
    TCN = 0;
    while (1) {
        for (a = 0; a < N; a++) {
            bestd[a] = 0;
        }
        for (a = 0; a < N; a++) {
            bestw[a][0] = 0;
        }
        scanf("%s", st1);
        for (a = 0; a < strlen(st1); a++) {
            st1[a] = toupper(st1[a]);
        }
        if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(stdin)) {
            if (TCN == 0) {
                TCN = 1;
            }
            if (QID != 0) {
                printf("ACCURACY TOP1: %.2f %% (%d / %d)\n", CCN / (float) TCN * 100, CCN, TCN);
                printf("Total accuracy: %.2f %%  Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %%",
                       CACN / (float) TACN * 100, SEAC / (float) SECN * 100, SYAC / (float) SYCN * 100);
            }
            QID++;
            scanf("%s", st1);
            if (feof(stdin)) {
                break;
            }
            printf("%s:\n", st1);
            TCN = 0;
            CCN = 0;
            continue;
        }
        if (!strcmp(st1, "EXIT")) {
            break;
        }
        scanf("%s", st2);
        for (a = 0; a < strlen(st2); a++) {
            st2[a] = toupper(st2[a]);
        }
        scanf("%s", st3);
        for (a = 0; a < strlen(st3); a++) {
            st3[a] = toupper(st3[a]);
        }
        scanf("%s", st4);
        for (a = 0; a < strlen(st4); a++) {
            st4[a] = toupper(st4[a]);
        }
        for (b = 0; b < words; b++) {
            if (!strcmp(&vocab[b * max_w], st1)) {
                break;
            }
        }
        b1 = b;
        for (b = 0; b < words; b++) {
            if (!strcmp(&vocab[b * max_w], st2)) {
                break;
            }
        }
        b2 = b;
        for (b = 0; b < words; b++) {
            if (!strcmp(&vocab[b * max_w], st3)) {
                break;
            }
        }
        b3 = b;
        for (a = 0; a < N; a++) {
            bestd[a] = 0;
            bestw[a][0] = 0;
        }
        TQ++;
        if (b1 == words) {
            continue;
        }
        if (b2 == words) {
            continue;
        }
        if (b3 == words) {
            continue;
        }
        for (b = 0; b < words; b++) {
            if (!strcmp(&vocab[b * max_w], st4)) {
                break;
            }
        }
        if (b == words) {
            continue;
        }
        for (a = 0; a < size; a++) {
            vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
        }
        TQS++;
        for (c = 0; c < words; c++) {
            if (c == b1) {
                continue;
            }
            if (c == b2) {
                continue;
            }
            if (c == b3) {
                continue;
            }
            dist = 0;
            for (a = 0; a < size; a++) {
                dist += vec[a] * M[a + c * size];
            }
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        if (!strcmp(st4, bestw[0])) {
            CCN++;
            CACN++;
            if (QID <= 5) {
                SEAC++;
            } else {
                SYAC++;
            }
        }
        if (QID <= 5) {
            SECN++;
        } else {
            SYCN++;
        }
        TCN++;
        TACN++;
    }
    printf("Questions seen / total: %d %d  %.2f %% \n", TQS, TQ, TQS / (float) TQ * 100);
    return 0;
}