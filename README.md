# Initialization
chi_k = 0
chi_k_minus_1 = 0
theta_k = 1

for k in range(K):

    # ---- Momentum update ----
    theta_k_plus_1 = (1 + sqrt(1 + 4 * theta_k**2)) / 2
    m_k = (theta_k - 1) / theta_k_plus_1

    # ---- Extrapolation step ----
    chi_hat = chi_k + m_k * (chi_k - chi_k_minus_1)

    # ---- Forward model ----
    phi_chi_hat = F_H( D * F(chi_hat) )

    # ---- Residual ----
    r_k = phi_chi_hat - y

    # ---- Gradient computation ----
    g_k = F_H( D * F(r_k) )

    # ---- Update step ----
    chi_next = chi_hat - (1 / L) * g_k

    # ---- Prepare for next iteration ----
    chi_k_minus_1 = chi_k
    chi_k = chi_next
    theta_k = theta_k_plus_1

# Output
return chi_k
