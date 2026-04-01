// サンプル C++ ファイル：意図的に複数の問題を含める
// コードレビュー実施テスト用

#include <vector>
#include <cmath>

class Matrix {
public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
    data_ = new double[rows * cols];
  }

  ~Matrix() {
    // メモリリーク：delete[] data_; を忘れた
  }

  double get(int i, int j) {
    return data_[i * cols_ + j];  // bounds チェックなし
  }

  void set(int i, int j, double value) {
    data_[i * cols_ + j] = value;
  }

private:
  double* data_;
  int rows_, cols_;
};

// 関数：ドキュメント不足
bool solve_system(const Matrix* A, const std::vector<double>* b,
                 std::vector<double>* x) {
  if (!A || !b) {
    return false;  // null 返却、例外処理なし
  }

  int n = b->size();
  for (int i = 0; i < n; ++i) {
    double sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += A->get(i, j) * (*b)[j];
    }
    x->push_back(sum);
  }
  return true;
}

// スタイル：不明確な引き数、型の implicit conversion
void compute_norm(const double data[], int size, double& result) {
  result = 0;
  for (int i = 0; i < size; ++i) {
    result += data[i] * data[i];  // overflow risk
  }
  result = sqrt(result);
}
