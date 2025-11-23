// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <math.h>
// #include <string.h>

// #include "quantization.h"
// #include "sparsity.h"
// #include "k_quantization.h"

// static void measure_metrics(const float *orig, const float *decomp, uint64_t N,
//                             double *mae, double *mse, double *max_abs) {
//     double m = 0.0, s = 0.0, mx = 0.0;
//     for (uint64_t i = 0; i < N; ++i) {
//         double e = (double)decomp[i] - (double)orig[i];
//         double ae = fabs(e);
//         m   += ae;
//         s   += e * e;
//         if (ae > mx) mx = ae;
//     }
//     *mae     = m / (double)N;
//     *mse     = s / (double)N;
//     *max_abs = mx;
// }

// /* Write decompressed data back to binary file in original format. */
// static int write_recovered_binary(const char *filename, uint8_t type, uint64_t n_embed, uint64_t n_tokens,
//                                   uint64_t tensor_size, const float *data) {
//     FILE *fp = fopen(filename, "wb");
//     if (!fp) {
//         fprintf(stderr, "Failed to open output file %s\n", filename);
//         return -1;
//     }
//     if (fwrite(&type, sizeof(uint8_t), 1, fp) != 1 ||
//         fwrite(&n_embed, sizeof(uint64_t), 1, fp) != 1 ||
//         fwrite(&n_tokens, sizeof(uint64_t), 1, fp) != 1 ||
//         fwrite(&tensor_size, sizeof(uint64_t), 1, fp) != 1) {
//         fprintf(stderr, "Failed to write header to %s\n", filename);
//         fclose(fp);
//         return -1;
//     }
//     if (fwrite(data, 1, tensor_size, fp) != tensor_size) {
//         fprintf(stderr, "Failed to write tensor data to %s\n", filename);
//         fclose(fp);
//         return -1;
//     }
//     fclose(fp);
//     return 0;
// }

// int main(void) {
//     const char *infile = "example/activation_112_3584.bin";
//     FILE *fp = fopen(infile, "rb");
//     if (!fp) {
//         fprintf(stderr, "Failed to open input file %s\n", infile);
//         return EXIT_FAILURE;
//     }

//     /* Read header */
//     uint8_t type;
//     uint64_t n_embed, n_tokens, tensor_size;
//     if (fread(&type, sizeof(uint8_t), 1, fp) != 1 ||
//         fread(&n_embed, sizeof(uint64_t), 1, fp) != 1 ||
//         fread(&n_tokens, sizeof(uint64_t), 1, fp) != 1 ||
//         fread(&tensor_size, sizeof(uint64_t), 1, fp) != 1) {
//         fprintf(stderr, "Failed to read header from %s\n", infile);
//         fclose(fp);
//         return EXIT_FAILURE;
//     }

//     if (type != 0) {
//         fprintf(stderr, "Unsupported element type: %u (expected 0 for FLOAT32)\n", type);
//         fclose(fp);
//         return EXIT_FAILURE;
//     }

//     uint64_t N = n_tokens * n_embed;
//     if (tensor_size != N * sizeof(float)) {
//         fprintf(stderr, "Tensor size mismatch: expected %lu bytes, got %lu\n",
//                 (unsigned long)(N * sizeof(float)), (unsigned long)tensor_size);
//         fclose(fp);
//         return EXIT_FAILURE;
//     }

//     float *orig = (float *)malloc(N * sizeof(float));
//     if (!orig) {
//         fprintf(stderr, "Failed to allocate original data buffer\n");
//         fclose(fp);
//         return EXIT_FAILURE;
//     }

//     if (fread(orig, sizeof(float), N, fp) != N) {
//         fprintf(stderr, "Failed to read tensor data from %s\n", infile);
//         free(orig);
//         fclose(fp);
//         return EXIT_FAILURE;
//     }
//     fclose(fp);

//     printf("Loaded real example: tokens=%lu, embed=%lu, N=%lu\n",
//            (unsigned long)n_tokens, (unsigned long)n_embed, (unsigned long)N);

//     /* Quantization: q8_0 and q4_0 */
//     {
//         const int qtypes[]   = {0 /*q8_0*/, 1 /*q4_0*/};
//         const char *qnames[] = {"q8_0", "q4_0"};

//         for (size_t i = 0; i < 2; ++i) {
//             const int qtype = qtypes[i];
//             const char *qname = qnames[i];
//             char outfile[64];
//             snprintf(outfile, sizeof(outfile), "%s.bin", qname);

//             quantized_array_t *qa = NULL;
//             if (quantize(orig, N, qtype, &qa) || !qa) {
//                 fprintf(stderr, "%s quantization failed\n", qname);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             float *rec = (float *)malloc(N * sizeof(float));
//             if (!rec) {
//                 fprintf(stderr, "Malloc failed for %s recovery buffer\n", qname);
//                 free_quantized_array(qa);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             if (dequantize(qa, rec)) {
//                 fprintf(stderr, "%s dequantization failed\n", qname);
//                 free(rec);
//                 free_quantized_array(qa);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double mae, mse, maxabs;
//             measure_metrics(orig, rec, N, &mae, &mse, &maxabs);

//             if (write_recovered_binary(outfile, type, n_embed, n_tokens, tensor_size, rec) != 0) {
//                 fprintf(stderr, "Failed to write %s\n", outfile);
//                 free(rec);
//                 free_quantized_array(qa);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double size_kb = get_quantized_array_size(qa) / 1024.0;
//             double bw = 8.0 * size_kb * 1024.0 / (double)N;
//             printf("   %s: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
//                    qname, size_kb, bw, mae, mse, maxabs);

//             free(rec);
//             free_quantized_array(qa);
//         }
//     }

//     /* Quantization: Q2_K (K-quant style, fp32 super params) */
//     {
//         if (N % WEIGHT_PER_SUPER_BLOCK != 0) {
//             /* Prevent out-of-bounds writes with current k_dequantize implementation. */
//             fprintf(stderr,
//                     "Q2_K skipped: N (%lu) must be a multiple of %d to be safe with current k_dequantize.\n",
//                     (unsigned long)N, WEIGHT_PER_SUPER_BLOCK);
//         } else {
//             const char *qname = "Q2_K";
//             const char *outfile = "q2_k.bin";

//             quantized_array_q2_k_t *qk = NULL;
//             if (k_quantize(orig, N, &qk) || !qk) {
//                 fprintf(stderr, "q2_k quantization failed\n");
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             float *rec = (float *)malloc(N * sizeof(float));
//             if (!rec) {
//                 fprintf(stderr, "Malloc failed for q2_k recovery buffer\n");
//                 free_quantized_q2_k_array(qk);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             if (k_dequantize(qk, rec)) {
//                 fprintf(stderr, "q2_k dequantization failed\n");
//                 free(rec);
//                 free_quantized_q2_k_array(qk);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double mae, mse, maxabs;
//             measure_metrics(orig, rec, N, &mae, &mse, &maxabs);

//             if (write_recovered_binary(outfile, type, n_embed, n_tokens, tensor_size, rec) != 0) {
//                 fprintf(stderr, "Failed to write %s\n", outfile);
//                 free(rec);
//                 free_quantized_q2_k_array(qk);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double size_kb = get_quantized_q2_k_array_size(qk) / 1024.0;
//             double bw = 8.0 * size_kb * 1024.0 / (double)N;
//             printf("   %s: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
//                    qname, size_kb, bw, mae, mse, maxabs);

//             free(rec);
//             free_quantized_q2_k_array(qk);
//         }
//     }

//     /* Sparsity variants */
//     {
//         const float ratios[] = {0.20f, 0.10f};
//         const char *rnames[] = {"sparse0.20", "sparse0.10"};

//         for (size_t i = 0; i < 2; ++i) {
//             float ratio = ratios[i];
//             const char *rname = rnames[i];
//             char outfile[64];
//             snprintf(outfile, sizeof(outfile), "%s.bin", rname);

//             sparse_array_t *sparse = NULL;
//             if (compress(orig, (uint16_t)n_tokens, (uint16_t)n_embed, ratio, &sparse)) {
//                 fprintf(stderr, "%s compression failed\n", rname);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             float *rec = (float *)malloc(N * sizeof(float));
//             if (!rec) {
//                 fprintf(stderr, "Malloc failed for %s recovery buffer\n", rname);
//                 free_sparse_array(sparse);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             if (decompress(sparse, rec)) {
//                 fprintf(stderr, "%s decompression failed\n", rname);
//                 free(rec);
//                 free_sparse_array(sparse);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double mae, mse, maxabs;
//             measure_metrics(orig, rec, N, &mae, &mse, &maxabs);

//             if (write_recovered_binary(outfile, type, n_embed, n_tokens, tensor_size, rec) != 0) {
//                 fprintf(stderr, "Failed to write %s\n", outfile);
//                 free(rec);
//                 free_sparse_array(sparse);
//                 free(orig);
//                 return EXIT_FAILURE;
//             }

//             double sparsity_actual = (double)sparse->num_sparse_features / (double)sparse->num_features;
//             double size_kb = get_sparse_array_size(sparse) / 1024.0;
//             double bw = 8.0 * size_kb * 1024.0 / (double)N;
//             printf("   %s: sparsity=%.3f, size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
//                    rname, sparsity_actual, size_kb, bw, mae, mse, maxabs);

//             free(rec);
//             free_sparse_array(sparse);
//         }
//     }

//     free(orig);
//     return EXIT_SUCCESS;
// }
