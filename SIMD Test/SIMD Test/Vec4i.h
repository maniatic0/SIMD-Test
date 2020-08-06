#pragma once

#include <stdint.h>
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.1
#include <tmmintrin.h> // SSE3
#include <cstring>

#define ALIGN(N) __declspec(align(N))
#define FORCE_INLINE __forceinline

union ALIGN(16) Vec4i
{
private:
	ALIGN(16) __m128i valsSIMD;
public:
	ALIGN(16) int32_t vals[4];
	struct
	{
		int32_t x;
		int32_t y;
		int32_t z;
		int32_t w;
	};
	struct
	{
		int32_t r;
		int32_t g;
		int32_t b;
		int32_t a;
	};

	/// <summary>
	/// New Zeroed Vector
	/// </summary>
	/// <returns>New Zeroed Vector</returns>
	inline Vec4i()
	{
		x = y = z = w = 0;
	}

	/// <summary>
	/// Set Vec4 from int arr
	/// </summary>
	/// <param name="_vals">Values to use</param>
	/// <returns>New Vec4i</returns>
	inline Vec4i(const int32_t _vals[4])
	{
		memcpy(vals, _vals, sizeof(int32_t ) * 4);
		//vals[0] = _vals[0];
		//vals[1] = _vals[1];
		//vals[2] = _vals[2];
		//vals[3] = _vals[3];
	}

	/// <summary>
	/// Set Vec4 from SIMD register
	/// </summary>
	/// <param name="_vals">Values to use</param>
	/// <returns>New Vec4i</returns>
	inline Vec4i(const __m128i &_vals)
	{
		valsSIMD = _vals;
	}

	/// <summary>
	/// Set Vec4 from SIMD register (Move constructor)
	/// </summary>
	/// <param name="_vals">Values to use</param>
	/// <returns>New Vec4i</returns>
	inline Vec4i(__m128i&& _vals)
	{
		valsSIMD = _vals;
	}

	/// <summary>
	/// Add Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator+(const Vec4i &rhs)
	{
		return Vec4i(_mm_add_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Add Two Vectors (Move Op)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator+(Vec4i&& rhs)
	{
		return Vec4i(_mm_add_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Add Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator+=(const Vec4i& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_add_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	/// <summary>
	/// Add Two Vectors (Move constructor)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator+=(Vec4i&& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_add_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	/// <summary>
	/// Subtracs Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator-(const Vec4i& rhs)
	{
		return Vec4i(_mm_sub_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Subtracs Two Vectors (Move Op)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator-(Vec4i&& rhs)
	{
		return Vec4i(_mm_sub_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Subtracs Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator-=(const Vec4i& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_sub_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	/// <summary>
	/// Subtracs Two Vectors (Move constructor)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator-=(Vec4i&& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_sub_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	/// <summary>
	/// Multiply Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator*(const Vec4i& rhs)
	{
		return Vec4i(_mm_mullo_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Multiply Two Vectors (Move Op)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i operator*(Vec4i&& rhs)
	{
		return Vec4i(_mm_mullo_epi32(this->valsSIMD, rhs.valsSIMD));
	}

	/// <summary>
	/// Multiply Two Vectors
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator*=(const Vec4i& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_mullo_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	/// <summary>
	/// Multiply Two Vectors (Move constructor)
	/// </summary>
	/// <param name="rhs">Right Hand Side Vector</param>
	/// <returns></returns>
	FORCE_INLINE Vec4i& operator*=(Vec4i&& rhs)
	{
		// MMM this is creating a temp somehow
		this->valsSIMD = _mm_mullo_epi32(this->valsSIMD, rhs.valsSIMD);
		return *this;
	}

	friend Vec4i abs(const Vec4i& vec);
	friend Vec4i abs(Vec4i&& vec);
	
};

/// <summary>
/// Get the Absolute Value per Component from a Vector
/// </summary>
/// <param name="vec">Vector</param>
/// <returns>Absolute Value per Component</returns>
FORCE_INLINE Vec4i abs(const Vec4i& vec)
{
	return Vec4i(_mm_abs_epi32(vec.valsSIMD));
}

/// <summary>
/// Get the Absolute Value per Component from a Vector
/// </summary>
/// <param name="vec">Vector</param>
/// <returns>Absolute Value per Component</returns>
FORCE_INLINE Vec4i abs(Vec4i&& vec)
{
	return Vec4i(_mm_abs_epi32(vec.valsSIMD));
}
