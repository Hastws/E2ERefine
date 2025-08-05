//
// Created by SENSETIME\quchunzhi on 25-6-24.
//

#ifndef INCLUDE_FUNCTIONS_AUTOALG_DISCRETE_ACTIVATION_H
#define INCLUDE_FUNCTIONS_AUTOALG_DISCRETE_ACTIVATION_H

#include "common/types.h"

namespace autoalg {
    inline Real BinaryActivation(const Real &z) {
        return z >= 0.0f ? 1.0f : 0.0f;
    }

    inline Real SignActivation(const Real &z) {
        if (z > 0.0f) return 1.0f;
        if (z < 0.0f) return -1.0f;
        return 0.0f;
    }
}
#endif // INCLUDE_FUNCTIONS_AUTOALG_DISCRETE_ACTIVATION_H
