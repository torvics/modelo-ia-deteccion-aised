#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vib_rf.h"
int main(void) {
  FILE *f = fopen("vectores_prueba_rf.csv", "r"); if (!f) return 1;
  static char line[8192]; static float feats[2000][VIB_N_FEAT]; static int sid[2000], isc[2000], py[2000]; int n = 0;
  while (fgets(line, sizeof line, f)) {
    char *t = strtok(line, ","); sid[n] = atoi(t); isc[n] = atoi(strtok(NULL, ","));
    for (int i = 0; i < VIB_N_FEAT; i++) feats[n][i] = (float)atof(strtok(NULL, ","));
    py[n] = atoi(strtok(NULL, ",\n")); n++;
  }
  int ok = 0, cur = -1; vib_calib_t c;
  for (int i = 0; i < n; i++) {
    if (sid[i] != cur) { cur = sid[i]; vib_calib_init(&c); for (int k = i; k < n && sid[k] == cur; k++) if (isc[k]) vib_calib_add(&c, feats[k]); }
    float p[3]; ok += (vib_predict(&c, feats[i], p) == py[i]);
  }
  printf("C vs scikit-learn (RF): coinciden %d de %d ventanas (%.2f%%)\n", ok, n, 100.0 * ok / n); return 0;
}
