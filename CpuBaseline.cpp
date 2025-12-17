#include "CpuBaseline.h"

static int32_t Get3DimArrayOff(int x, int y, int z, int xsize, int ysize, int zsize)
{
    int32_t off = 0;

    off = z * xsize * ysize + y * xsize + x;

    return off;
}

static std::complex<float>* RangeAddWin_CalFft(int16_t* inBuf, int xsize, int ysize, int zsize, int nfft)
{
    int x, y, z;
    int srcoff, dstoff;
    fftwf_complex* in, * out;
    fftwf_plan p;
    std::complex<float>* pin;
    std::complex<float>* pout;

    std::complex<float>* outBuf = (std::complex<float>*)malloc(nfft * ysize * zsize * sizeof(std::complex<float>));
    std::vector<float> winBuf;

    create_symmetric_hanning_window(winBuf, xsize);

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);

    p = fftwf_plan_dft_1d(nfft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (z = 0; z < zsize; z++)
    {
        for (y = 0; y < ysize; y++)
        {
            pin = (std::complex<float>*)in;
            pout = (std::complex<float>*)out;

            for (x = 0; x < xsize; x++)
            {
                srcoff = Get3DimArrayOff(x, y, z, xsize, ysize, zsize);
                pin[x] = (std::complex<float>)(winBuf[x] * inBuf[srcoff]);
            }
            for (; x < nfft; x++)
            {
                pin[x] = 0;
            }
            fftwf_execute(p);
            for (x = 0; x < nfft; x++)
            {
                dstoff = Get3DimArrayOff(x, y, z, nfft, ysize, zsize);
                outBuf[dstoff] = pout[x];
            }
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return outBuf;
}

static std::complex<float>* Fft1D_DiscardHalf(std::complex<float>* inBuf, int xsize, int ysize, int zsize)
{
    int x, y, z;
    int srcoff, dstoff;
    std::complex<float>* outBuf = (std::complex<float>*)malloc(xsize / 2 * ysize * zsize * sizeof(std::complex<float>));

    for (z = 0; z < zsize; z++)
    {
        for (y = 0; y < ysize; y++)
        {
            for (x = 0; x < xsize / 2; x++)
            {
                srcoff = Get3DimArrayOff(x, y, z, xsize, ysize, zsize);
                dstoff = Get3DimArrayOff(x, y, z, xsize / 2, ysize, zsize);
                outBuf[dstoff] = inBuf[srcoff];
            }
        }
    }

    return outBuf;
}

static std::complex<float>* fft1d_func(std::vector<int16_t>& indata, int fft_loops)
{
    std::complex<float>* fftwin;
    std::complex<float>* fft1d;
    int nfft = 512;

    fftwin = RangeAddWin_CalFft(indata.data(), adcNum, chirpNum, rxNum, nfft);
    fft1d = Fft1D_DiscardHalf(fftwin, nfft, chirpNum, rxNum);
    free(fftwin);

    return fft1d;
}

std::complex<float>* fft2d_func(std::complex<float>* fft1d, int fft_loops)
{

    int nfft = chirpNum;
    int x, y, z;
    int xsize = adcNum / 2, ysize = chirpNum, zsize = fft_loops;
    int srcoff, dstoff;

    //** if mod(cfg.num_chirp, 3) == 0
    //**         NFFT_2d = 2^nextpow2(cfg.num_chirp/3) * 3;  % last stage of FFT is x3
    //**     else
    //**     NFFT_2d = 2^nextpow2(cfg.num_chirp);
    //** end

    fftwf_complex* in, * out;
    fftwf_plan p;
    std::complex<float>* pin;
    std::complex<float>* pout;

    std::complex<float>* fft2d = (std::complex<float>*)malloc(xsize * nfft * zsize * sizeof(std::complex<float>));
    std::vector<float> winBuf;

    create_symmetric_hanning_window(winBuf, ysize);

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
    p = fftwf_plan_dft_1d(nfft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (z = 0; z < zsize; z++)
    {
        for (x = 0; x < xsize; x++)
        {
            pin = (std::complex<float>*)in;
            pout = (std::complex<float>*)out;
            for (y = 0; y < ysize; y++)
            {
                srcoff = Get3DimArrayOff(x, y, z, xsize, ysize, zsize);
                pin[y] = winBuf[y] * fft1d[srcoff];

                //				in[x][0] = crealf(winBuf[y + 1] * inBuf[srcoff]);
                //				in[x][1] = cimagf(winBuf[y + 1] * inBuf[srcoff]);
            }
            for (; y < nfft; y++)
            {
                pin[y] = 0;
                //				in[x][0] = 0;
                //				in[x][1] = 0;
            }
            fftwf_execute(p); // 执行变换
            for (y = 0; y < nfft; y++)
            {
                dstoff = Get3DimArrayOff(x, y, z, xsize, nfft, zsize);
                fft2d[dstoff] = pout[y];
                //				fft2d[dstoff] = CMPLXF(out[y][0], out[y][1]);
            }
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return fft2d;
}

void cpu_process_old_way(std::vector<int16_t>& indata, std::vector<std::complex<float>>& outdata)
{
    std::complex<float>* fft1d, * fft2d;

    std::vector<int16_t> adc_ddm;
    adc_ddm.resize(chirpNum * adcNum * rxNum);
    for (int i = 0; i < chirpNum * adcNum * rxNum; i++)
    {
        adc_ddm[i] = indata[i] - 2048;
    }

    fft1d = fft1d_func(adc_ddm, rxNum);
    fft2d = fft2d_func(fft1d, rxNum);

    int out_x = adcNum / 2;
    int out_y = chirpNum;
    int out_z = rxNum;
    size_t output_count = (size_t)out_x * out_y * out_z;

    if (outdata.size() != output_count) {
        outdata.resize(output_count);
    }

    if (fft2d != nullptr) {
        memcpy(outdata.data(), fft2d, output_count * sizeof(std::complex<float>));
    }

    if (fft1d) free(fft1d);
    if (fft2d) free(fft2d);
}

