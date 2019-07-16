#pragma once

#include <vector>

namespace RetroWarp
{
template <typename T>
class Canvas
{
public:
	void resize(unsigned width_, unsigned height_)
	{
		width = width_;
		height = height_;
		data.clear();
		data.resize(width * height);
	}

	T &get(unsigned x, unsigned y)
	{
		return data[y * width + x];
	}

	const T &get(unsigned x, unsigned y) const
	{
		return data[y * width + x];
	}

	unsigned get_width() const
	{
		return width;
	}

	unsigned get_height() const
	{
		return height;
	}

	const T *get_data() const
	{
		return data.data();
	}

private:
	std::vector<T> data;
	unsigned width = 0;
	unsigned height = 0;
};
}
