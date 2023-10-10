package main

import (
	"math"
)

func getAngle(x, y float64) int64 {
	tg := y / x
	if x != 0 {
		tg = 999
	}
	if y > 0 {
		if x > 0 {
			if tg < 0.414 {
				return 2
			}
			if tg > 2.414 {
				return 4
			} else {
				return 3
			}
		} else {
			if tg < -2.414 {
				return 4
			}
			if tg < -0.414 {
				return 5
			} else if tg > -0.414 {
				return 6
			}
		}
	} else {
		if x > 0 {
			if tg < -2.414 {
				return 0
			}
			if tg < -0.414 {
				return 1
			}
			if tg > -0.414 {
				return 2
			}
		} else {
			if tg > 2.414 {
				return 0
			}
			if tg < 2.414 {
				return 7
			}
			if tg < 0.414 {
				return 6
			}
		}
	}
	return 0
}

func roll(img [][]float64, kernel [][]float64) [][]float64 {
	size := len(kernel)
	s := size / 2
	matr := make([][]float64, len(img))
	for i := range img {
		matr[i] = make([]float64, len(img[i]))
		copy(matr[i], img[i])
	}
	for i := s; i < len(matr)-s; i++ {
		for j := s; j < len(matr[i])-s; j++ {
			val := 0.0
			for k := -s; k <= s; k++ {
				for l := -s; l <= s; l++ {
					val += img[i+k][j+l] * kernel[k+s][l+s]
				}
			}
			matr[i][j] = val
		}
	}
	return matr
}

func main() {
	Gx := [][]float64{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
	Gy := [][]float64{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}
	img := [][]float64{}
	img_Gx := roll(img, Gx)
	img_Gy := roll(img, Gy)
	matr_grd_length := make([][]float64, len(img))
	for i := range img {
		matr_grd_length[i] = make([]float64, len(img[i]))
		for j := range img[i] {
			matr_grd_length[i][j] = math.Sqrt(img_Gx[i][j]*img_Gx[i][j] + img_Gy[i][j]*img_Gy[i][j])
		}
	}
	matr_grd_dir := make([][]int64, len(img))
	for i := range img {
		matr_grd_dir[i] = make([]int64, len(img[i]))
	}
	for i := 0; i < len(img); i++ {
		matr_grd_dir[i] = make([]int64, len(img[0]))
	}
	for i := 0; i < len(img); i++ {
		for j := 0; j < len(img[0]); j++ {
			matr_grd_dir[i][j] = getAngle(img_Gx[i][j], img_Gy[i][j])
		}
	}

	border := make([][]float64, len(img))
	for i := 0; i < len(img); i++ {
		border[i] = make([]float64, len(img[0]))
		copy(border[i], img[i])
	}

	for i := 0; i < len(img); i++ {
		for j := 0; j < len(img[0]); j++ {
			grad := matr_grd_length[i][j]
			direct := matr_grd_dir[i][j]
			if i == 0 || i == len(img)-1 || j == 0 || j == len(img[0])-1 {
				border[i][j] = 0
			} else {
				var x_shift, y_shift int
				if direct == 0 || direct == 4 {
					x_shift = 0
				} else if direct > 0 && direct < 4 {
					x_shift = 1
				} else {
					x_shift = -1
				}
				if direct == 2 || direct == 6 {
					y_shift = 0
				} else if direct > 2 && direct < 6 {
					y_shift = -1
				} else {
					y_shift = 1
				}
				if grad >= matr_grd_length[i+y_shift][j+x_shift] && grad >= matr_grd_length[i-y_shift][j-x_shift] {
					border[i][j] = 255
				} else {
					border[i][j] = 0
				}
			}
		}
	}
}
