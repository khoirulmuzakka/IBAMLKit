#include "pileup.h" 
#include <atomic>
#include "conv.h"
#include <iostream>


// rebin: accepts numpy arrays (float64, contiguous). No implicit conversions => no elementwise copying.
namespace {
    // Precompute 1/width for old bins and validate monotonicity.
    inline std::vector<double> compute_inv_widths_vec(const double* old_edges, size_t n_old) {
        std::vector<double> invw(n_old);
        for (size_t i = 0; i < n_old; ++i) {
            const double w = old_edges[i+1] - old_edges[i];
            if (w <= 0.0) throw std::invalid_argument("bin_edges_old must be strictly increasing");
            invw[i] = 1.0 / w;
        }
        return invw;
    }

    // Core rebin for a single spectrum using pointer sweep and precomputed inv widths.
    inline void rebin_with_inv(const double* old_edges, const double* new_edges,
                               size_t n_old, size_t n_new,
                               const double* inv_widths,
                               const double* spectrum,
                               double* out_row) {
        size_t i_old = 0;
        for (size_t k = 0; k < n_new; ++k) {
            const double left_new = new_edges[k];
            const double right_new = new_edges[k+1];
            double sum = 0.0;

            while (i_old < n_old && old_edges[i_old+1] <= left_new) ++i_old;
            size_t j = i_old;
            while (j < n_old && old_edges[j] < right_new) {
                const double left_overlap = max(old_edges[j], left_new);
                const double right_overlap = min(old_edges[j+1], right_new);
                if (right_overlap > left_overlap) {
                    sum += (spectrum[j] * inv_widths[j]) * (right_overlap - left_overlap);
                }
                ++j;
            }
            out_row[k] = sum;
        }
    }
}

py::array_t<double> rebin_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges_old,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges_new,
    py::array_t<double, py::array::c_style | py::array::forcecast> spectrum)
{
    // --- shapes and pointers ---
    const size_t n_old_edges = bin_edges_old.size();
    const size_t n_new_edges = bin_edges_new.size();
    if (n_old_edges < 2 || n_new_edges < 2)
        throw std::invalid_argument("Not enough bin edges provided.");
    const size_t n_old = n_old_edges - 1;
    const size_t n_new = n_new_edges - 1;

    if (spectrum.size() != n_old)
        throw std::invalid_argument("spectrum length must be len(bin_edges_old)-1");

    const double* old = static_cast<const double*>(bin_edges_old.data());
    const double* ne  = static_cast<const double*>(bin_edges_new.data());
    const double* spec = static_cast<const double*>(spectrum.data());

    // Precompute inv widths once
    const std::vector<double> invw = compute_inv_widths_vec(old, n_old);

    // Allocate and compute
    py::array_t<double> result(n_new);
    double* out = static_cast<double*>(result.request().ptr);
    rebin_with_inv(old, ne, n_old, n_new, invw.data(), spec, out);
    return result;
}


py::array_t<double> fast_pileup_batch_numpy(
    const py::array_t<double>& spectra,      // (n_spectra, K)
    const py::array_t<double>& real_times,  // (n_spectra,)
    const py::array_t<double>& live_times,
    const py::array_t<double>& fudge_factors, // (n_spectra,)
    bool clip_negative )
{
    auto spec_buf = spectra.request();
    if (spec_buf.ndim != 2) throw std::invalid_argument("spectra must be 2D");
    const size_t n_spec = static_cast<size_t>(spec_buf.shape[0]);
    const size_t K = static_cast<size_t>(spec_buf.shape[1]);
    const double* spec_ptr = static_cast<const double*>(spec_buf.ptr);

    const double* rt_ptr = static_cast<const double*>(real_times.request().ptr);
    const double* lt_ptr = static_cast<const double*>(live_times.request().ptr);
    const double* ff_ptr = static_cast<const double*>(fudge_factors.request().ptr);
    if (real_times.size() != n_spec || fudge_factors.size() != n_spec)
        throw std::invalid_argument("real_times and fudge_factors must match spectra.shape[0]");

    // Validate inputs before parallel region (exceptions inside OpenMP are problematic)
    for (size_t s=0; s<n_spec; ++s) {
        if (rt_ptr[s] <= 0.0) throw std::invalid_argument("real_time must be >0");
    }

    size_t out_len = 2*K-1;
    py::array_t<double> out({n_spec, out_len});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    py::gil_scoped_release release; // heavy computation outside GIL

    // Parallelize across spectra with per-thread FFT workspace
    #ifdef _OPENMP
    #pragma omp parallel if(n_spec > 1)
    {
        std::vector<fftconv::cplx> fft_buffer; // thread-local reusable FFT buffer
        std::vector<double> conv(out_len);     // thread-local real buffer
        #pragma omp for schedule(static)
        for (long long s_ll = 0; s_ll < static_cast<long long>(n_spec); ++s_ll) {
            const size_t s = static_cast<size_t>(s_ll);
            const double* spec = spec_ptr + s*K;
            double N = 0.0;
            for (size_t i = 0; i < K; ++i) N += spec[i];
            const double real_time = rt_ptr[s];
            const double live_time = lt_ptr[s];
            double ratio= live_time/real_time;
            const double fudge = ff_ptr[s];
            const double alpha = N/real_time;
            const double A = (fudge/real_time) * std::exp(-alpha*fudge);

            fftconv::convolve_fft_self_workspace(spec, K, conv.data(), fft_buffer);

            double* out_row = out_ptr + s*out_len;
            for (size_t i = 0; i < K; ++i) {
                const double spec_i = spec[i];
                double val = spec_i - 2.0*A*N*spec_i + A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
            for (size_t i = K; i < out_len; ++i) {
                double val = A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
        }
    }
    #else
    {
        std::vector<fftconv::cplx> fft_buffer;
        std::vector<double> conv(out_len);
        for (size_t s = 0; s < n_spec; ++s) {
            const double* spec = spec_ptr + s*K;
            double N = std::accumulate(spec, spec+K, 0.0);
            const double real_time = rt_ptr[s];
            const double live_time = lt_ptr[s];
            double ratio= live_time/real_time;
            const double fudge = ff_ptr[s];
            const double alpha = N/real_time;
            const double A = (fudge/real_time) * std::exp(-alpha*fudge);
            fftconv::convolve_fft_self_workspace(spec, K, conv.data(), fft_buffer);
            double* out_row = out_ptr + s*out_len;
            for (size_t i = 0; i < K; ++i) {
                const double spec_i = spec[i];
                double val = spec_i - 2.0*A*N*spec_i + A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
            for (size_t i = K; i < out_len; ++i) {
                double val = A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
        }
    }
    #endif

    return out;
}


py::array_t<double> convert_to_channel_space_and_pileup_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> c,
    py::array_t<double, py::array::c_style | py::array::forcecast> real_times,
    py::array_t<double, py::array::c_style | py::array::forcecast> live_times,
    py::array_t<double, py::array::c_style | py::array::forcecast> fudge_factors,
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> E_space_spectra,
    bool clip_negative)
{
    auto spec_buf = E_space_spectra.request();
    if (spec_buf.ndim != 2) throw std::invalid_argument("E_space_spectra must be 2D");
    const size_t n_spec = static_cast<size_t>(spec_buf.shape[0]);
    const size_t K = static_cast<size_t>(spec_buf.shape[1]);
    const double* e_spec_all = static_cast<const double*>(spec_buf.ptr);

    if (a.size() != n_spec || b.size() != n_spec || c.size() != n_spec)
        throw std::invalid_argument("a, b, c must match number of spectra");
    if (real_times.size() != n_spec || fudge_factors.size() != n_spec || live_times.size() != n_spec || r.size() != n_spec)
        throw std::invalid_argument("real_times, live_times, fudge_factors, r must match number of spectra");

    const double* a_ptr = static_cast<const double*>(a.data());
    const double* b_ptr = static_cast<const double*>(b.data());
    const double* c_ptr = static_cast<const double*>(c.data());
    const double* rt_ptr = static_cast<const double*>(real_times.data());
    const double* lt_ptr = static_cast<const double*>(live_times.data());
    const double* ff_ptr = static_cast<const double*>(fudge_factors.data());
    const double* r_ptr  = static_cast<const double*>(r.data());

    // Validate real_times up front
    for (size_t s = 0; s < n_spec; ++s) if (rt_ptr[s] <= 0.0) throw std::invalid_argument("real_time must be >0");

    // Determine per-spectrum channel count M[s] by inverting calibration at Emax (old upper edge)
    std::vector<size_t> M(n_spec, K);
    size_t Mmax = 1;
    const double Emax = static_cast<double>(K);
    for (size_t s = 0; s < n_spec; ++s) {
        const double aa = a_ptr[s];
        const double bb = b_ptr[s];
        const double cc = c_ptr[s];
        double ch_est = 0.0;
        if (std::abs(cc) < 1e-18) {
            if (bb > 0.0) ch_est = (Emax - aa) / bb; else ch_est = 0.0;
        } else {
            const double A = cc;
            const double B = bb;
            const double C = aa - Emax;
            const double disc = B*B - 4.0*A*C;
            if (disc > 0.0) {
                const double sqrtD = std::sqrt(disc);
                const double r1 = (-B + sqrtD) / (2.0*A);
                const double r2 = (-B - sqrtD) / (2.0*A);
                ch_est = max(r1, r2);
            } else {
                ch_est = 0.0;
            }
        }
        size_t m = (ch_est > 0.0 ? static_cast<size_t>(std::ceil(ch_est)) : static_cast<size_t>(1));
        if (m < 1) m = 1;
        M[s] = m;
        if (m > Mmax) Mmax = m;
    }

    const size_t out_len = 2*Mmax - 1;
    py::array_t<double> out({n_spec, out_len});
    double* out_all = static_cast<double*>(out.request().ptr);

    // Prebuild uniform old edges [0, 1, ..., K]
    std::vector<double> old_edges(K + 1);
    for (size_t i = 0; i <= K; ++i) old_edges[i] = static_cast<double>(i);
    // inv widths for old edges (all 1.0 since width=1)
    std::vector<double> invw_old(K, 1.0);

    py::gil_scoped_release release;

    std::atomic<bool> invalid_calib(false);

    #ifdef _OPENMP
    #pragma omp parallel if(n_spec > 1)
    {
        std::vector<double> new_edges;
        std::vector<double> spec_ch;
        std::vector<double> conv;
        std::vector<fftconv::cplx> fft_buffer; // thread-local reusable FFT buffer

        #pragma omp for schedule(static)
        for (long long s_ll = 0; s_ll < static_cast<long long>(n_spec); ++s_ll) {
            const size_t s = static_cast<size_t>(s_ll);
            const double* spec_e = e_spec_all + s*K;

            const double aa = a_ptr[s];
            const double bb = b_ptr[s];
            const double cc = c_ptr[s];
            const size_t Ms = M[s];
            const size_t out_len_s = 2*Ms - 1;

            // Build new edges in energy for channel bins via E(ch) = a + b*ch + c*ch^2
            new_edges.resize(Ms + 1);
            bool ok = true;
            for (size_t ch = 0; ch <= Ms; ++ch) new_edges[ch] = aa + bb*static_cast<double>(ch) + cc*static_cast<double>(ch)*static_cast<double>(ch);
            // Ensure strictly increasing; if not, mark invalid
            for (size_t i = 1; i <= Ms; ++i) {
                if (!(new_edges[i] > new_edges[i-1])) { ok = false; break; }
            }
            if (!ok) {
                // Fill this row with zeros and continue; will raise after loop
                double* out_row = out_all + s*out_len;
                std::fill(out_row, out_row + out_len, 0.0);
                invalid_calib.store(true, std::memory_order_relaxed);
                continue;
            }

            // Rebin E-space spectrum (old_edges) into channel bins (new_edges)
            spec_ch.assign(Ms, 0.0);
            rebin_with_inv(old_edges.data(), new_edges.data(), K, Ms, invw_old.data(), spec_e, spec_ch.data());

            // Apply per-spectrum normalization factor r before pileup
            const double rscale = r_ptr[s];
            if (rscale != 1.0) for (size_t i = 0; i < Ms; ++i) spec_ch[i] *= rscale;

            // Pileup on channel spectrum using reusable FFT workspace
            conv.assign(out_len_s, 0.0);
            fftconv::convolve_fft_self_workspace(spec_ch.data(), Ms, conv.data(), fft_buffer);

            const double N = std::accumulate(spec_ch.begin(), spec_ch.end(), 0.0);
            const double real_time = rt_ptr[s];
            const double live_time = lt_ptr[s];
            double ratio= live_time/real_time;
            const double fudge = ff_ptr[s];
            const double alpha = (real_time > 0.0 ? N/real_time : 0.0);
            const double A = (real_time > 0.0 ? (fudge/real_time) * std::exp(-alpha*fudge) : 0.0);

            double* out_row = out_all + s*out_len;
            std::fill(out_row, out_row + out_len, 0.0);
            for (size_t i = 0; i < out_len_s; ++i) {
                const double spec_i = (i < Ms ? spec_ch[i] : 0.0);
                double val = spec_i - 2.0*A*N*spec_i + A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
        }
    }
    #else
    {
        std::vector<double> new_edges;
        std::vector<double> spec_ch;
        std::vector<double> conv;
        std::vector<fftconv::cplx> fft_buffer; // reusable buffer
        for (size_t s = 0; s < n_spec; ++s) {
            const double* spec_e = e_spec_all + s*K;

            const double aa = a_ptr[s];
            const double bb = b_ptr[s];
            const double cc = c_ptr[s];

            const size_t Ms = M[s];
            const size_t out_len_s = 2*Ms - 1;

            new_edges.resize(Ms + 1);
            bool ok = true;
            for (size_t ch = 0; ch <= Ms; ++ch) new_edges[ch] = aa + bb*static_cast<double>(ch) + cc*static_cast<double>(ch)*static_cast<double>(ch);
            for (size_t i = 1; i <= Ms; ++i) {
                if (!(new_edges[i] > new_edges[i-1])) { ok = false; break; }
            }
            if (!ok) {
                double* out_row = out_all + s*out_len;
                std::fill(out_row, out_row + out_len, 0.0);
                invalid_calib.store(true, std::memory_order_relaxed);
                continue;
            }

            spec_ch.assign(Ms, 0.0);
            rebin_with_inv(old_edges.data(), new_edges.data(), K, Ms, invw_old.data(), spec_e, spec_ch.data());

            // Apply per-spectrum normalization
            const double rscale = r_ptr[s];
            if (rscale != 1.0) for (size_t i = 0; i < Ms; ++i) spec_ch[i] *= rscale;

            conv.assign(out_len_s, 0.0);
            fftconv::convolve_fft_self_workspace(spec_ch.data(), Ms, conv.data(), fft_buffer);

            const double N = std::accumulate(spec_ch.begin(), spec_ch.end(), 0.0);
            const double real_time = rt_ptr[s];
            const double live_time = lt_ptr[s];
            double ratio= live_time/real_time;

            const double* rt_ptr = static_cast<const double*>(real_times.data());
            const double fudge = ff_ptr[s];
            const double alpha = (real_time > 0.0 ? N/real_time : 0.0);
            const double A = (real_time > 0.0 ? (fudge/real_time) * std::exp(-alpha*fudge) : 0.0);

            double* out_row = out_all + s*out_len;
            std::fill(out_row, out_row + out_len, 0.0);
            for (size_t i = 0; i < out_len_s; ++i) {
                const double spec_i = (i < Ms ? spec_ch[i] : 0.0);
                double val = spec_i - 2.0*A*N*spec_i + A*conv[i];
                if (clip_negative && val < 0.0) val = 0.0;
                out_row[i] = ratio*val;
            }
        }
    }
    #endif

    if (invalid_calib.load(std::memory_order_relaxed))
        throw std::invalid_argument("Calibration produces non-increasing channel energy edges for at least one spectrum");

    return out;
}
