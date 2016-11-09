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
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(-1);
    }
    for (a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        a = AddWordToVocab(word);
        fscanf(fin, "%lld\n", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in training file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(-1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
//    a = posix_memalign((void **) &syn0, 128, vocab_size * layer1_size * sizeof(real));
    syn0 = _aligned_malloc(128, vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {
        printf("Memory allocation failed\n");
        exit(-1);
    }
    if (hs) {
        syn1 = _aligned_malloc(128, vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {
            printf("Memory allocation failed\n");
            exit(-1);
        }
        for (a = 0; a < vocab_size; a++) {
            for (b = 0; b < layer1_size; b++) {
                syn1[a * layer1_size + b] = 0;
            }
        }
    }
    if (negative > 0) {
        syn1neg = _aligned_malloc(128, vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {
            printf("Memory allocation failed\n");
            exit(-1);
        }
        for (a = 0; a < vocab_size; a++) {
            for (b = 0; b < layer1_size; b++) {
                syn1neg[a * layer1_size + b] = 0;
            }
        }
    }
    for (a = 0; a < vocab_size; a++) {
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        }
    }
    CreateBinaryTree();

}

void *TrainModelThread(void *id) {

}

void TrainModel() { }

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) {
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\tSet threshold for occurence of words. Thoes that appear with high frequency in the training data\n");
        printf("\twill be randomly down-sampled; default is 1e-3, useful range is (0,1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0(not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10(0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vector are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0(off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constrcted from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1(use 0 for skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0)layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)cbow = atoi(argv[i + 1]);
    if (cbow) alpha = 0.05;
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0)strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0)window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-hs", argc, argv)) > 0)hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-classes", argc, argv)) > 0)classes = atoi(argv[i + 1]);

    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
    expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; ++i) {
        expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // pre computer the exp table
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }
    TrainModel();
    return 0;
}













