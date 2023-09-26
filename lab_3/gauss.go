package main

import (
	"math"
)

func get_gauss(x, y, sigma, a, b int) float64 {

	exp := math.Exp(-(math.Pow(float64(x-a), 2) + math.Pow(float64(y-b), 2)) / (2 * math.Pow(float64(sigma), 2)))
	return float64(1) / (2 * math.Pi * math.Pow(float64(sigma), 2)) * exp
}

func gaussian_blur(img [][]float64, size, deviation int) [][]float64 {
	kernel := make([][]float64, size)
	for i := range kernel {
		kernel[i] = make([]float64, size)
	}
	a, b := (size+1)/2, (size+1)/2
	for i := range kernel {
		for j := range kernel[i] {
			kernel[i][j] = get_gauss(i, j, deviation, a, b)
		}
	}
	sum := 0.0
	for i := range kernel {
		for j := range kernel[i] {
			sum += kernel[i][j]
		}
	}
	for i := range kernel {
		for j := range kernel[i] {
			kernel[i][j] /= sum
		}

	}
	blur := [][]float64{}
	copy(blur, img)
	s := size / 2
	for i := s; i < len(blur)-s; i++ {
		for j := s; j < len(blur[0])-s; j++ {
			var value float64
			for k := -s; k <= s; k++ {
				for l := -s; l <= s; l++ {
					value += blur[i+k][j+l] * kernel[s+l][s+k]
				}
			}
			blur[i][j] = value
		}
	}
	return blur

}
