#include "image.cuh"
#include <vector>
#include <png.h>
#include <cstdlib>
#include <cstdarg>
#include <cstring>

Image::Image(uint width, uint height) : width_(width), height_(height) {
    pixels = new RGBColor[width * height];
}

Image::Image() : width_(0), height_(0), pixels(nullptr) {}

Image::Image(const Image &other) : width_(0), height_(0), pixels(nullptr) {
    *this = other;
}

void Image::destroy() {
    if (pixels)
        delete[] pixels;
    width_ = 0;
    height_ = 0;
    pixels = nullptr;
}

void Image::create(uint width, uint height) {
    destroy();
    width_ = width;
    height_ = height;
    pixels = new RGBColor[width * height];
}

Image::~Image() {
    destroy();
}

RGBColor &Image::operator()(uint x, uint y) {
    /*
    assert(x < width_ && y < height_)
        << "Coordinates [" << x << "," << y << "] are out of bounds of an image [" << width_ << "," << height_
        << "]";
    */
    return pixels[y * width_ + x];
}

const RGBColor &Image::operator()(uint x, uint y) const {
    /*
    assert(x < width_ && y < height_)
        << "Coordinates [" << x << "," << y << "] are out of bounds of an image [" << width_ << "," << height_
        << "]";
        */
    return pixels[y * width_ + x];
}

void Image::clear(const RGBColor &color) {
    for (uint i = 0; i < width_ * height_; ++i)
        pixels[i] = color;
}

void abort_(const char *s, ...) {
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void Image::writePNG(const std::string &_fileName) {
    std::vector<png_byte> byteData(width_ * height_ * 3);
    std::vector<png_byte>::iterator ptr = byteData.begin();
    for (size_t i = 0; i < width_ * height_; ++i) {
        RGBColor v = pixels[i].clamp();
        *ptr++ = (unsigned char)(v.r * 255);
        *ptr++ = (unsigned char)(v.g * 255);
        *ptr++ = (unsigned char)(v.b * 255);
    }

    std::vector<png_byte *> rowData(height_);
    for (int i = 0; i < height_; i++)
        rowData[i] = i * width_ * 3 + &byteData.front();

    /* create file */
    FILE *fp = fopen(_fileName.c_str(), "wb");
    if (!fp)
        abort_("[write_png_file] File %s could not be opened for writing", _fileName.c_str());

    /* initialize stuff */
    png_structp png_ptr;
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[write_png_file] png_create_write_struct failed");

    png_infop info_ptr;
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[write_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during init_io");

    png_init_io(png_ptr, fp);

    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing header");

    png_set_IHDR(png_ptr, info_ptr, width_, height_, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing bytes");

    png_write_image(png_ptr, (png_byte **)&rowData.front());

    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during end of write");

    png_write_end(png_ptr, NULL);

    fclose(fp);
}

void Image::readPNG(const std::string &filename) {
    png_byte header[8]; // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp)
        abort_("[read_png_file] File %s could not be opened for reading", filename.c_str());
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
        abort_("[read_png_file] File %s is not recognized as a PNG file", filename.c_str());

    /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[read_png_file] png_create_read_struct failed");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    create(png_get_image_width(png_ptr, info_ptr), png_get_image_height(png_ptr, info_ptr));

    int number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    std::vector<png_byte> byteData(rowbytes * height_);
    std::vector<png_byte *> rowData(height_);
    for (int i = 0; i < height_; i++)
        rowData[i] = i * rowbytes + &byteData.front();

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during read_image");

    png_read_image(png_ptr, &rowData.front());

    fclose(fp);

    RGBColor *it = pixels;
    for (size_t y = 0; y < height_; y++) {
        png_byte *b = rowData[y];
        for (size_t x = 0; x < width_; x++) {
            RGBColor &v = *it++;
            v.r = (float)(*b++) / 255.f;
            v.g = (float)(*b++) / 255.f;
            v.b = (float)(*b++) / 255.f;
            if (png_get_channels(png_ptr, info_ptr) == 4)
                b++;
        }
    }
}

Image &Image::operator=(const Image &other) {
    if (other.pixels) {
        create(other.width(), other.height());
        memcpy(pixels, other.pixels, width_ * height_ * sizeof(RGBColor));
    } else
        destroy();
    return *this;
}
