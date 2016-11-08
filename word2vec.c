//
// Created by BaoQiang on 2016/11/8.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGH 1000
#define MAX_CODE_LENGH 40

const int vocab_hash_size = 30000000;

typedef float real;

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *) malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; ++a) {
        train_words_pow += pow(vocab[a].cn, power);
    }
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < vocab_size; ++a) {
        table[a] = i;
        if (a / (double) table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) {
            i = vocab_size - 1;
        }
    }
}

void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) {
            continue;
        }
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') {
                    ungetc(ch, fin);
                }
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            } else {
                continue;
            }
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) {
            a--;
        }
    }
    word[a] = 0;
}

int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); ++a) {
        hash = hash * 257 + word[a];
    }
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) {
            return -1;
        }
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) {
            return vocab_hash[hash];
        }
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) {
        return -1;
    }
    return SearchVocab(word);
}

int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) {
        length = MAX_STRING;
    }
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
//    reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) {
        hash = (hash + 1) % vocab_hash_size;
    }
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

void SortVocab() {
    int a, size;
    unsigned int hash;
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_size; ++a) {
        vocab_hash[a] = -1;
    }
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; ++a) {
        if ((vocab[a].cn < min_count) && (a != 0)) {
            vocab_size--;
            free(vocab[a].word);
        } else {
            hash = GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) {
                hash = (hash + 1) % vocab_hash_size;
            }
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
//    binary tree construction
    for (a = 0; a < vocab_size; ++a) {
        vocab[a].code = (char *) calloc(MAX_CODE_LENGH, sizeof(char));
        vocab[a].point = (int *) calloc(MAX_CODE_LENGH, sizeof(int));
    }
}

void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) {
        if (vocab[a].cn > min_reduce) {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        } else {
            free(vocab[a].word);
        }
    }
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    for (a = 0; a < vocab_size; a++) {
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) {
            hash = (hash + 1) % vocab_hash_size;
        }
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// create binary huffman tree using the word counts
// frequent words will have short uniques binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGH];
    char code[MAX_CODE_LENGH];
    long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) {
        count[a] = vocab[a].cn;
    }
    for (a = vocab_size; a < vocab_size * 2; a++) {
        count[a] = 1e15;
    }
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
//    construct huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
//        min1i
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        //        min2i
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
//    now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) {
                break;
            }
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }

    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(-1);
    }
    vocab_size = 0;
    AddWordToVocab((char *) "</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else {
            vocab[a].cn++;
        }
        if (vocab_size > vocab_hash_size * 0.7) {
            ReduceVocab();
        }
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocad() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) {
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    }
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
}
























