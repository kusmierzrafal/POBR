#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <stack>

/* ───────── selektory koloru ───────── */
inline bool isRedHard(int B, int G, int R) {
    int maxC = std::max({ R,G,B }), minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return sat > 60 && R > 120 && (R - G > 60) && (R - B > 60) &&
        (double)R / (G + 1) > 1.6 && (double)R / (B + 1) > 1.6;
}
inline bool isRedSoft(int B, int G, int R) {
    int maxC = std::max({ R,G,B }), minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return sat > 40 && R > 100 && (R - G > 40) && (R - B > 40) &&
        (double)R / (G + 1) > 1.4 && (double)R / (B + 1) > 1.4;
}
/* ─ Yellow: twardy & miękki ─ */
inline bool isYellowHard(int B, int G, int R) {
    int maxC = std::max({ R,G,B }), minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return R > 150 && G > 150 && B < 130 && std::abs(R - G) < 60 && sat > 45;
}
inline bool isYellowSoft(int B, int G, int R) {
    int maxC = std::max({ R,G,B }), minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return R > 110 && G > 110 && B < 150 && std::abs(R - G) < 80 && sat > 25;
}
/* ─ Blue litery ─ */
inline bool isBlue(int B, int G, int R) {
    int maxC = std::max({ R,G,B }), minC = std::min({ R,G,B });
    int sat = maxC - minC;
    return B > 90 && (B - R > 35) && (B - G > 35) && sat > 35;
}

/* ───── struktura flood-filla ───── */
struct Comp { int minR = 1e9, minC = 1e9, maxR = -1, maxC = -1, area = 0; };

int main()
{
    cv::Mat src = cv::imread("lidl_logo_1.png");            /* ← zmień nazwę pliku */
    if (!src.data) { std::cerr << "Błąd wczytania pliku\n"; return -1; }
    if (src.channels() != 3 || src.depth() != 0) {
        std::cerr << "Obraz nie jest 8-bit BGR\n"; return -1;
    }
    const int rows = src.rows, cols = src.cols;

    /* ─── maski ─── */
    cv::Mat redMask(rows, cols, src.depth()),
        used(rows, cols, src.depth()),
        finalRed(rows, cols, src.depth()),
        yellowMask(rows, cols, src.depth()),
        blueMask(rows, cols, src.depth());

    for (int r = 0;r < rows;++r)
        for (int c = 0;c < cols;++c) {
            redMask.at<uchar>(r, c) = used.at<uchar>(r, c) = 0;
            finalRed.at<uchar>(r, c) = yellowMask.at<uchar>(r, c) = 0;
            blueMask.at<uchar>(r, c) = 0;
        }

    /* ─── 1. Czerwień: twardy + miękki ─── */
    for (int r = 0;r < rows;++r) {
        const uchar* row = src.ptr<uchar>(r);
        for (int c = 0;c < cols;++c) {
            int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
            if (isRedHard(B, G, R)) redMask.at<uchar>(r, c) = 255;
        }
    }
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

    /* ─── 2. Flood-fill CCL po czerwonej masce ─── */
    std::vector<Comp> comps;
    auto inside = [&](int r, int c) {return r >= 0 && r < rows && c >= 0 && c < cols;};
    for (int r = 0;r < rows;++r)
        for (int c = 0;c < cols;++c)
            if (redMask.at<uchar>(r, c) && !used.at<uchar>(r, c)) {
                Comp cp; std::stack<cv::Point> st; st.push({ c,r });
                while (!st.empty()) {
                    cv::Point p = st.top(); st.pop();
                    int y = p.y, x = p.x;
                    if (!inside(y, x) || used.at<uchar>(y, x) || !redMask.at<uchar>(y, x)) continue;
                    used.at<uchar>(y, x) = 1; ++cp.area;
                    cp.minR = std::min(cp.minR, y); cp.maxR = std::max(cp.maxR, y);
                    cp.minC = std::min(cp.minC, x); cp.maxC = std::max(cp.maxC, x);
                    st.push({ x + 1,y }); st.push({ x - 1,y });
                    st.push({ x,y + 1 }); st.push({ x,y - 1 });
                }
                comps.push_back(cp);
            }

    /* ─── 3. wybieramy pierścienie ─── */
    std::vector<int> rings;
    for (size_t i = 0;i < comps.size();++i) {
        int w = comps[i].maxC - comps[i].minC + 1,
            h = comps[i].maxR - comps[i].minR + 1;
        if (w < 40 || h < 40) continue;
        double asp = (double)std::max(w, h) / std::min(w, h);
        if (asp > 2.5) continue;
        double fill = (double)comps[i].area / (w * h);
        if (fill > 0.25) continue;
        rings.push_back((int)i);
    }

    /* ─── 4. Żółty (twardy + miękki) + niebieskie litery ─── */
    for (int idx : rings) {
        const Comp& ring = comps[idx];

        /* 4a. kopiuj pierścień */
        for (int r = ring.minR;r <= ring.maxR;++r)
            for (int c = ring.minC;c <= ring.maxC;++c)
                if (redMask.at<uchar>(r, c))
                    finalRed.at<uchar>(r, c) = 255;

        /* 4b. twarde żółte piksele */
        for (int r = ring.minR + 1;r <= ring.maxR - 1;++r) {
            const uchar* row = src.ptr<uchar>(r);
            for (int c = ring.minC + 1;c <= ring.maxC - 1;++c) {
                int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
                if (isYellowHard(B, G, R))
                    yellowMask.at<uchar>(r, c) = 255;
            }
        }
        /* 4c. miękkie żółte przy brzegu istniejącej żółci */
        for (int r = ring.minR + 2;r <= ring.maxR - 2;++r) {
            const uchar* row = src.ptr<uchar>(r);
            for (int c = ring.minC + 2;c <= ring.maxC - 2;++c) {
                if (yellowMask.at<uchar>(r, c)) continue;
                int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
                if (!isYellowSoft(B, G, R)) continue;
                bool near = false;
                for (int dy = -1;dy <= 1 && !near;++dy)
                    for (int dx = -1;dx <= 1 && !near;++dx)
                        if (yellowMask.at<uchar>(r + dy, c + dx)) near = true;
                if (near) yellowMask.at<uchar>(r, c) = 255;
            }
        }

        /* 4d. niebieskie litery wewnątrz BB */
        for (int r = ring.minR + 1;r <= ring.maxR - 1;++r) {
            const uchar* row = src.ptr<uchar>(r);
            for (int c = ring.minC + 1;c <= ring.maxC - 1;++c) {
                if (finalRed.at<uchar>(r, c) || yellowMask.at<uchar>(r, c)) continue;
                int B = row[3 * c], G = row[3 * c + 1], R = row[3 * c + 2];
                if (isBlue(B, G, R)) blueMask.at<uchar>(r, c) = 255;
            }
        }
    }

    /* ─── 5. podgląd best for now─── */
    cv::imshow("Oryginał", src);
    cv::imshow("Czerwony pierścień+kropka", finalRed);
    cv::imshow("Żółte tło", yellowMask);
    cv::imshow("Niebieskie litery", blueMask);
    cv::waitKey(-1);
    return 0;
}
