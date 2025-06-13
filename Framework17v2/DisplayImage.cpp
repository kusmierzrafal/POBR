#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    std::cout << "Start ..." << std::endl;

    std::string image_name = "lidl_logo_1.png";

    cv::Mat image = cv::imread(image_name);

    cv::imshow(image_name, image);
    

    cv::waitKey(-1);
    return 0;
}
