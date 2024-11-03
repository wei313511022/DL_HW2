#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(){
    // 用來管理動態記憶體的指標
  int *dynArr;

  // 指定空間大小
  long int arrLen = 100000000000;

  // 取得記憶體空間
  dynArr = (int*)malloc( arrLen * sizeof(int) );

  if( dynArr == NULL ) {
    // 無法取得記憶體空間
    fprintf(stderr, "Error: unable to allocate required memory\n");
    return 1;
  }

  // 使用動態取得的記憶體空間
  int i;
  for (i = 0; i < arrLen; ++i) {
    dynArr[i] = i;
    printf("%d ", dynArr[i]);
  }

  // 釋放記憶體空間
  free(dynArr);

  return 0;

}