#include "base64_utils.hpp"
#include <algorithm>

namespace Base64Utils {
    
    static const std::string base64Chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    static inline bool isBase64(unsigned char c) {
        return (isalnum(c) || (c == '+') || (c == '/'));
    }

    std::string encode(const unsigned char* data, size_t len) {
        std::string ret;
        int i = 0;
        int j = 0;
        unsigned char charArray3[3];
        unsigned char charArray4[4];

        while (len--) {
            charArray3[i++] = *(data++);
            if (i == 3) {
                charArray4[0] = (charArray3[0] & 0xfc) >> 2;
                charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
                charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
                charArray4[3] = charArray3[2] & 0x3f;

                for(i = 0; (i <4) ; i++)
                    ret += base64Chars[charArray4[i]];
                i = 0;
            }
        }

        if (i) {
            for(j = i; j < 3; j++)
                charArray3[j] = '\0';

            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            for (j = 0; (j < i + 1); j++)
                ret += base64Chars[charArray4[j]];

            while((i++ < 3))
                ret += '=';
        }

        return ret;
    }

    std::string encodeMat(const cv::Mat& mat) {
        std::vector<uchar> buffer;
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compressionParams.push_back(9);

        if (!cv::imencode(".png", mat, buffer, compressionParams)) {
            return "";
        }

        return encode(buffer.data(), buffer.size());
    }

    std::vector<unsigned char> decode(const std::string& encodedString) {
        int inLen = encodedString.size();
        int i = 0;
        int j = 0;
        int inIdx = 0;
        unsigned char charArray4[4], charArray3[3];
        std::vector<unsigned char> ret;

        while (inLen-- && ( encodedString[inIdx] != '=') && isBase64(encodedString[inIdx])) {
            charArray4[i++] = encodedString[inIdx]; inIdx++;
            if (i ==4) {
                for (i = 0; i <4; i++)
                    charArray4[i] = base64Chars.find(charArray4[i]);

                charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
                charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
                charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

                for (i = 0; (i < 3); i++)
                    ret.push_back(charArray3[i]);
                i = 0;
            }
        }

        if (i) {
            for (j = i; j <4; j++)
                charArray4[j] = 0;

            for (j = 0; j <4; j++)
                charArray4[j] = base64Chars.find(charArray4[j]);

            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            for (j = 0; (j < i - 1); j++) ret.push_back(charArray3[j]);
        }

        return ret;
    }
}