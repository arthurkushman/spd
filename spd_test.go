package spd

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestMatrixSPD_Decompose(t *testing.T) {
	data := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	spd := NewMatrixSPD(0.001, 5000)
	U, V := spd.Decompose(data, 2)

	rU, cU := U.Caps()
	rV, cV := V.Caps()
	require.Equal(t, cV, rU)
	require.Equal(t, rV, cU)
}
