#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stack>

/* ───────── selektory koloru ───────── */
inline bool isRedHard(const cv::Vec3b& p)
{
    int B = p[0], G = p[1], R = p[2];
    int maxC = std::max({ R, G, B });
    int minC = std::min({ R, G, B });
    int sat = maxC - minC;
    return sat > 60 && R > 120 &&
        (R - G > 60) && (R - B > 60) &&
        (double)R / (G + 1) > 1.6 &&
        (double)R / (B + 1) > 1.6;
}

inline bool isRedSoft(const cv::Vec3b& p)
{
    int B = p[0], G = p[1], R = p[2];
    int maxC = std::max({ R, G, B });
    int minC = std::min({ R, G, B });
    int sat = maxC - minC;
    return sat > 40 && R > 100 &&
        (R - G > 40) && (R - B > 40) &&
        (double)R / (G + 1) > 1.4 &&
        (double)R / (B + 1) > 1.4;
}

/* ───────── struktura pomocnicza ───────── */
struct Comp {
    int minR = 1e9, minC = 1e9, maxR = -1, maxC = -1, area = 0;
};

int main()
{
    /* 1. wczytanie */
    cv::Mat src = cv::imread("lidl_logo_3.png");      // ← zmień nazwę pliku, jeśli trzeba
    if (!src.data) { std::cerr << "Błąd wczytania pliku\n"; return -1; }

    /* 2. maska czerwieni (0/255) */
    cv::Mat redMask(src.rows, src.cols, CV_8U);
    for (int r = 0; r < src.rows; ++r)
        for (int c = 0; c < src.cols; ++c)
            redMask.at<uchar>(r, c) = 0;

    /* 2a. twardy selektor */
    for (int r = 0; r < src.rows; ++r)
        for (int c = 0; c < src.cols; ++c)
            if (isRedHard(src.at<cv::Vec3b>(r, c)))
                redMask.at<uchar>(r, c) = 255;

    /* 2b. miękki selektor przy brzegu */
    for (int r = 1; r < src.rows - 1; ++r)
        for (int c = 1; c < src.cols - 1; ++c)
            if (!redMask.at<uchar>(r, c) &&
                isRedSoft(src.at<cv::Vec3b>(r, c)))
            {
                bool near = false;
                for (int dy = -1; dy <= 1 && !near; ++dy)
                    for (int dx = -1; dx <= 1 && !near; ++dx)
                        if (redMask.at<uchar>(r + dy, c + dx)) near = true;
                if (near) redMask.at<uchar>(r, c) = 255;
            }

    /* 3. własny CCL (flood-fill 4-sąsiedztwo) */
    cv::Mat used(src.rows, src.cols, CV_8U);
    for (int r = 0; r < src.rows; ++r)
        for (int c = 0; c < src.cols; ++c)
            used.at<uchar>(r, c) = 0;

    std::vector<Comp> comps;
    auto in = [&](int r, int c)
        { return r >= 0 && r < src.rows && c >= 0 && c < src.cols; };

    for (int r = 0; r < src.rows; ++r)
        for (int c = 0; c < src.cols; ++c)
            if (redMask.at<uchar>(r, c) && !used.at<uchar>(r, c))
            {
                Comp cp;
                std::stack<cv::Point> st; st.push({ c, r });
                while (!st.empty())
                {
                    cv::Point p = st.top(); st.pop();
                    int y = p.y, x = p.x;
                    if (!in(y, x) || used.at<uchar>(y, x) || !redMask.at<uchar>(y, x))
                        continue;
                    used.at<uchar>(y, x) = 1;
                    ++cp.area;
                    cp.minR = std::min(cp.minR, y);
                    cp.maxR = std::max(cp.maxR, y);
                    cp.minC = std::min(cp.minC, x);
                    cp.maxC = std::max(cp.maxC, x);
                    st.push({ x + 1, y }); st.push({ x - 1, y });
                    st.push({ x, y + 1 }); st.push({ x, y - 1 });
                }
                comps.push_back(cp);
            }

    /* 4. ── FILTR PIERŚCIENI ── (poprawiony) */
    std::vector<int> keepIdx;
    for (size_t i = 0; i < comps.size(); ++i)
    {
        int w = comps[i].maxC - comps[i].minC + 1;
        int h = comps[i].maxR - comps[i].minR + 1;
        if (w < 40 || h < 40) continue;          // zbyt małe

        double aspect = (double)std::max(w, h) / std::min(w, h);
        if (aspect > 2.5) continue;              // pozwalamy na przekoszenie

        double fill = (double)comps[i].area / (w * h);
        if (fill > 0.25) continue;               // pierścień musi być pustawy

        keepIdx.push_back((int)i);
    }

    /* 5. składamy wynik: pierścienie + zawartość środka */
    cv::Mat finalMask(src.rows, src.cols, CV_8U);
    for (int r = 0; r < src.rows; ++r)
        for (int c = 0; c < src.cols; ++c)
            finalMask.at<uchar>(r, c) = 0;

    for (int idx : keepIdx)
    {
        const Comp& ring = comps[idx];
        // kopiujemy sam pierścień
        for (int r = ring.minR; r <= ring.maxR; ++r)
            for (int c = ring.minC; c <= ring.maxC; ++c)
                if (redMask.at<uchar>(r, c))
                    finalMask.at<uchar>(r, c) = 255;

        // kopiujemy małe czerwone wyspy całkowicie wewnątrz pierścienia
        for (size_t j = 0; j < comps.size(); ++j)
        {
            if ((int)j == idx) continue;
            const Comp& small = comps[j];
            if (small.minR >= ring.minR && small.maxR <= ring.maxR &&
                small.minC >= ring.minC && small.maxC <= ring.maxC)
            {
                for (int r = small.minR; r <= small.maxR; ++r)
                    for (int c = small.minC; c <= small.maxC; ++c)
                        if (redMask.at<uchar>(r, c))
                            finalMask.at<uchar>(r, c) = 255;
            }
        }
    }

    /* 6. podgląd / zapis */
    cv::imshow("Oryginał", src);
    cv::imshow("Tylko ramka + kropka", finalMask);
    // cv::imwrite("red_clean.png", finalMask);

    cv::waitKey(-1);
    return 0;
}
