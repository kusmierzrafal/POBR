#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stack>

/* ───────── selektory koloru ───────── */
inline bool isRedHard(int B, int G, int R)
{
    int maxC = std::max({ R,G,B });
    int minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return sat > 60 && R > 120 &&
        (R - G > 60) && (R - B > 60) &&
        (double)R / (G + 1) > 1.6 &&
        (double)R / (B + 1) > 1.6;
}
inline bool isRedSoft(int B, int G, int R)
{
    int maxC = std::max({ R,G,B });
    int minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return sat > 40 && R > 100 &&
        (R - G > 40) && (R - B > 40) &&
        (double)R / (G + 1) > 1.4 &&
        (double)R / (B + 1) > 1.4;
}

/* ───────── struktura pomocnicza ───────── */
struct Comp { int minR = 1e9, minC = 1e9, maxR = -1, maxC = -1, area = 0; };

int main()
{
    /* 1. wczytanie */
    cv::Mat src = cv::imread("lidl_logo_1.png");          // ← zmień nazwę pliku
    if (!src.data) { std::cerr << "Błąd wczytania pliku\n"; return -1; }

    if (src.channels() != 3 || src.depth() != 0) {              // 0 ⇒ 8-bit
        std::cerr << "Obraz nie jest 8-bitowy 3-kanałowy\n";
        return -1;
    }

    const int rows = src.rows, cols = src.cols;

    /* 2. maski pomocnicze (1-kanałowe, 8-bit) */
    cv::Mat redMask(rows, cols, src.depth());            // depth()==0
    cv::Mat used(rows, cols, src.depth());
    cv::Mat finalMask(rows, cols, src.depth());

    for (int r = 0;r < rows;++r)
        for (int c = 0;c < cols;++c) {
            redMask.at<uchar>(r, c) = 0;
            used.at<uchar>(r, c) = 0;
            finalMask.at<uchar>(r, c) = 0;
        }

    /* 2a. twardy selektor */
    for (int r = 0;r < rows;++r) {
        const uchar* row = src.ptr<uchar>(r);
        for (int c = 0;c < cols;++c) {
            int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
            if (isRedHard(B, G, R)) redMask.at<uchar>(r, c) = 255;
        }
    }

    /* 2b. miękki selektor przy brzegu */
    for (int r = 1;r < rows - 1;++r) {
        const uchar* row = src.ptr<uchar>(r);
        for (int c = 1;c < cols - 1;++c) {
            if (redMask.at<uchar>(r, c)) continue;
            int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
            if (!isRedSoft(B, G, R)) continue;

            bool near = false;
            for (int dy = -1;dy <= 1 && !near;++dy)
                for (int dx = -1;dx <= 1 && !near;++dx)
                    if (redMask.at<uchar>(r + dy, c + dx)) near = true;

            if (near) redMask.at<uchar>(r, c) = 255;
        }
    }

    /* 3. własny CCL (4-sąsiedztwo) */
    std::vector<Comp> comps;
    auto inside = [&](int r, int c) { return r >= 0 && r < rows && c >= 0 && c < cols; };

    for (int r = 0;r < rows;++r)
        for (int c = 0;c < cols;++c)
            if (redMask.at<uchar>(r, c) && !used.at<uchar>(r, c)) {
                Comp cp;
                std::stack<cv::Point> st; st.push({ c,r });
                while (!st.empty()) {
                    cv::Point p = st.top(); st.pop();
                    int y = p.y, x = p.x;
                    if (!inside(y, x) || used.at<uchar>(y, x) || !redMask.at<uchar>(y, x))
                        continue;
                    used.at<uchar>(y, x) = 1;
                    ++cp.area;
                    cp.minR = std::min(cp.minR, y); cp.maxR = std::max(cp.maxR, y);
                    cp.minC = std::min(cp.minC, x); cp.maxC = std::max(cp.maxC, x);
                    st.push({ x + 1,y }); st.push({ x - 1,y });
                    st.push({ x,y + 1 }); st.push({ x,y - 1 });
                }
                comps.push_back(cp);
            }

    /* 4. filtr pierścieni (tolerancja przekoszenia) */
    std::vector<int> keepIdx;
    for (size_t i = 0;i < comps.size();++i) {
        int w = comps[i].maxC - comps[i].minC + 1;
        int h = comps[i].maxR - comps[i].minR + 1;
        if (w < 40 || h < 40) continue;
        double aspect = (double)std::max(w, h) / std::min(w, h);
        if (aspect > 2.5) continue;
        double fill = (double)comps[i].area / (w * h);
        if (fill > 0.25) continue;
        keepIdx.push_back((int)i);
    }

    /* 5. złożenie wyniku: pierścienie + środek */
    for (int idx : keepIdx) {
        const Comp& ring = comps[idx];
        for (int r = ring.minR;r <= ring.maxR;++r)
            for (int c = ring.minC;c <= ring.maxC;++c)
                if (redMask.at<uchar>(r, c))
                    finalMask.at<uchar>(r, c) = 255;

        for (size_t j = 0;j < comps.size();++j) {
            if ((int)j == idx) continue;
            const Comp& small = comps[j];
            if (small.minR >= ring.minR && small.maxR <= ring.maxR &&
                small.minC >= ring.minC && small.maxC <= ring.maxC) {
                for (int r = small.minR;r <= small.maxR;++r)
                    for (int c = small.minC;c <= small.maxC;++c)
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
