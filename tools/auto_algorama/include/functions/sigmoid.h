#ifndef INCLUDE_AUTOALG_SIGMOD_H
#define INCLUDE_AUTOALG_SIGMOD_H
#include "common/types.h"

namespace autoalg {
    inline Real Sigmoid(const Real &v) {
        return 1.0 / (1.0 + std::exp(-v));
    }

    inline Real Probit(const Real &z) {
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }

    inline Real Tanh(const Real &v) {
        // return (std::exp(v) - std::exp(-v)) / (std::exp(v) + std::exp(-v));
        // return 2 * Sigmoid(2 * v) - 1;
        return std::tanh(v);
    }

    // Shifted and Scaled Sigmoid (SSS)
    inline Real ShiftedScaledSigmoid(const Real &z) {
        const Real a = 0.2;
        const Real b = 6.0;
        return 1.0 / (1.0 + std::exp(-a * (z - b)));
    }

    // Variant Sigmoid Function (VSF)
    inline Real VariantSigmoid(const Real &z) {
        const Real a = 1.0;
        const Real b = 5.0;
        const Real c = 0.5;
        return a / (1.0 + std::exp(-b * z)) - c;
    }

    // Scaled Hyperbolic Tangent (stanh)
    inline Real ScaledTanh(const Real &z) {
        const Real a = 1.7159;
        const Real b = 2.0 / 3.0 * 23.0;
        return a * std::tanh(b * z);
    }

    // Bi-modal Sigmoid
    inline Real BimodalSigmoid(const Real &z) {
        const Real b = 2.0;
        return 0.5 * (
                   1.0 / (1.0 + std::exp(-z)) +
                   1.0 / (1.0 + std::exp(-z - b))
               );
    }
}

#endif // INCLUDE_AUTOALG_SIGMOD_H
