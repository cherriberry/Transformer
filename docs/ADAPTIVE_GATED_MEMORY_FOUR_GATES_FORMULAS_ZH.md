写入门：
$$
g_i^{\mathrm{write}}=\sigma\!\left(W_{\mathrm{write}}\bar{h}_i+b_{\mathrm{write}}\right)
$$

更新门：
$$
g_{i,r}^{\mathrm{update}}=\sigma\!\left(W_{\mathrm{update}}\left[v_i^{\mathrm{cand}};v_r^{\mathrm{old}};s_{i,r}\right]+b_{\mathrm{update}}\right)
$$

保留门：
\[
$$g_r^{\mathrm{ret}}=\sigma\!\left(W_{\mathrm{ret}}\left[k_r;v_r;m_r\right]+b_{\mathrm{ret}}\right)$$
\]

融合门：
\[
$$g_t^{\mathrm{fusion}}=\sigma\!\left(W_{\mathrm{fusion}}\left[h_t;c_t\right]+b_{\mathrm{fusion}}\right)$$
\]
