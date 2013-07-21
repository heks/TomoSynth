/*****************************************************************************/
/*MDStart
	(C) Copyright 2008 Triple Ring Technologies, Incorporated
	All Rights Reserved

	THIS IS THE UNPUBLISHED PROPRIETARY SOURCE CODE OF TRIPLE RING TECHNOLOGIES.
	The copyright notice above does not evidence any actual or intended
	publication of such source code.

	   Description:	useful 'C' mnemonics and types

	Revision History:	1.1		22 Jan 1999		A.P. Lowell
							added 'xInt64' types to use the underlying
							compiler data type if available

						1.0		30 Jan 1998		A.P. Lowell
							Initial release
MDEnd*/
/*IDStart*/
#ifndef		stdhdr_h
#define		stdhdr_h
/*IDEnd*/


	/*** includes ***/

		/*** standard 'C' libraries ***/
		#ifdef _WIN32
#include <assert.h>			/* assertions for debugging (ANSI) */
        #endif
#include <limits.h>			/* integer limits (ANSI) */
#include <string.h>			/* String support ( strlen, strncpy, strcpy,...) */
#include <stdio.h>			/* printf support */
#include <float.h>			/* floating-point limtis (ANSI) */

	/*
		NOTE:	The following construction is specific to Microsoft Visual C++.
				It must be adjusted for other compilers.

				Note that use of complex numbers as an inherent type is only
				supported within C++ programs.
	*/
#ifdef		__cplusplus
	#ifdef _WIN32
		#pragma warning ( disable: 4275 )
		#include <complex>
		#pragma warning ( default: 4275 )
	#else	// GCC
		#include <complex>
	#endif
#endif		/* __cplusplus */

		/*** extended 'C' libraries ***/

		/*** third-party libraries ***/

		/*** common headers and libraries ***/

		/*** project headers and libraries ***/

		/*** module headers and libraries ***/

		/*** local headers ***/


	/*** mnemonic definitions ***/

		/*** mnemonics for data types ***/
			/*** integer types ***/
#define SNativeInt_MIN			INT_MIN
#define SNativeInt_MAX			INT_MAX
#define UNativeInt_MAX			UINT_MAX

#define SInt8_MIN				SCHAR_MIN
#define SInt8_MAX				SCHAR_MAX
#define UInt8_MAX				UCHAR_MAX

#define SInt16_MIN				SHRT_MIN
#define SInt16_MAX 				SHRT_MAX
#define UInt16_MAX				USHRT_MAX

#define SInt32_MIN 				LONG_MIN
#define SInt32_MAX 				LONG_MAX
#define UInt32_MAX				ULONG_MAX

#define SInt64_MIN				( ((SInt64)1) << 63 )
#define SInt64_MAX				( -( SInt64_MIN + 1 ) )
#define UInt64_MAX				( ~(UInt64)0 )


		/*** floating-point types ***/
#define FloatSP_DIG 			FLT_DIG
#define FloatSP_MIN	 			FLT_MIN
#define FloatSP_MAX				FLT_MAX
#define FloatSP_EPSILON			FLT_EPSILON

#define FloatDP_DIG				DBL_DIG
#define FloatDP_MIN				DBL_MIN
#define FloatDP_MAX				DBL_MAX
#define FloatDP_EPSILON			DBL_EPSILON

			/*** OS handles ***/
#define _INVALID_HANDLE			((_Handle)-1U)
#define _NULL_HANDLE			((_Handle)0)

			/*** Pointer values ***/
#if !defined ( _NULL )
	#ifdef __cplusplus
		#define _NULL			0
	#else
		#define _NULL			((void*)0)
	#endif
#endif


			/*** logical state types ***/
				/*** boolean logic values (_lBool) ***/
					/*
						NOTE:	For the following I specifically did NOT use the
								standard 'C++' type 'bool' and its corresponding
								constants 'true' and 'false' because I wanted a
								type that would be consistent across 'C' and
								'C++' platforms.
					*/
#define _FALSE					( (_lBool)( 0 != 0 ) )
#define _TRUE					( (_lBool)( 0 == 0 ) )

#define _LOW					_FALSE
#define _HIGH					_TRUE
#define _CLEAR					_FALSE
#define _RESET					_FALSE
#define _SET					_TRUE
#define _DEASSERTED				_FALSE
#define _ASSERTED				_TRUE

				/*** switch states (_lBool, _State, or _LState) ***/
#define _OFF 					_FALSE
#define _ON						_TRUE
#define _OPEN					_FALSE
#define _CLOSED					_TRUE
#define _INACTIVE				_FALSE
#define _ACTIVE					_TRUE
#define _UNSIGNALLED			_FALSE
#define _SIGNALLED				_TRUE

				/*** decision states ***/
				/*** (_Decision, _LDecision, _State, or _LState) ***/
#define _NO						0
#define _MAYBE					(-1)
#define _YES					1
#define _FAIL					_NO
#define _PASS					_YES
#define _NOTOK					_FAIL
#define _OK						_PASS
#define _BAD					_NO
#define _INDIFFERENT			_MAYBE
#define _GOOD					_YES
#define _NOGO					_NO
#define _STOP					_NO
#define _GO						_YES
#define _INVALID				_NO
#define _VALID					_YES

				/*** numbers (_State) ***/
#define _NEGATIVE				(-1)
#define _MINUS					_NEGATIVE
#define _ZERO					0
#define _POSITIVE				1
#define _PLUS					_POSITIVE

				/*** result states (_Result or _LResult) ***/
#define _FAILURE				(-1)	/* note: add new failure codes < -1 */
#define _SUCCESS				0		/* note: add new success codes > 0 */


	/*** macro definitions ***/


		/*** useful utilities ***/

#define _min(a,b)					( ( (a) < (b) ) ? (a) : (b) )
#define _max(a,b)					( ( (a) > (b) ) ? (a) : (b) )
#define _sign(a)					( ( (a) > 0 ) \
										?	_POSITIVE \
										:	( ( (a) < 0 ) \
												?	_NEGATIVE \
												:	_ZERO ) )


		/*** status testing ***/

/*FHStart
			 Macro:	_FAILED(result)
					_SUCCEEDED(result)

					_CONTINUE(decision)
					_DISCONTINUE(decision)

	   Description:	tests for status mnemonics

					use as follows:

						_Result		Func1();
						_LResult		Func2();

						FuncResult1 = Func1();
						FuncResult2 = Func2();

						if ( _FAILED ( FuncResult1 ) ) {
							'Func1' failed;
						}

						if ( _SUCCEEDED ( FuncResult2 ) ) {
							'Func2' succeeded;
						}

						_Decision	Func3();
						_Decision	Func4();

						FuncResult3 = Func3();
						FuncResult4 = Func4();

						if ( _CONTINUE ( FuncResult3 ) ) {
							'Func3' decision was affirmative
								(_YES, _OK, _GOOD, etc.)
						}

						if ( _DISCONTINUE ( FuncResult4 ) ) {
							'Func4' decision was negative
								(_NO, _NOTOK, _BAD, etc.)
						}

		 Arguments:	_Result or _LResult value for _SUCCEEDED/_FAILED
						_Decision or _LDecision value for _CONTINUE/_DISCONTINUE
  Global Variables:	none
		   Returns:	_TRUE or _FALSE

	   Constraints:	none
FHEnd*/
#define _FAILED(result)				( (result) <= _FAILURE )
#define _SUCCEEDED(result)			( (result) >= _SUCCESS )


#define _CONTINUE(decision)		( (decision) >= _YES )
#define _DISCONTINUE(decision)	( (decision) <= _NO )


		/*** bit manipulations ***/

#if		_YES

/*FHStart
			 Macro:	_BitSet()
					_BitClear()

	   Description:	determines the state of the specified bit

		 Arguments:	<integer type> 'data' to be tested
					<positive integer> 'bit' number
						(0 <= bit <= bit size of Data)
  Global Variables:	none
		   Returns:	_TRUE if bit is in the specified state, _FALSE otherwise;
					bit 0 is LSB

	   Constraints:	none
FHEnd*/
#define _BitSet(data,bit) 		( ( (data) & ( 1 << (bit) ) ) != 0 )
#define _BitClear(data,bit)		( !_BitSet ( data, bit ) )

/*FHStart
			 Macro:	_AnyBitsSet()
					_AllBitsSet()
					_AnyBitsClear()
					_AllBitsClear()

	   Description:	determines the state of the specified bits

		 Arguments:	<integer type> 'data' to be tested
					<integer type> bit 'mask'
						(mask is same size as data)
  Global Variables:	none
		   Returns:	_TRUE if bit(s) corresponding to those SET (1) in the
					mask are in specified state, _FALSE otherwise;

	   Constraints:	none
FHEnd*/
#define _AnyBitsSet(data,mask)	( ( (data) & (mask) ) != 0 )
#define _AllBitsSet(data,mask)	( ( (data) & (mask) ) == (mask) )
#define _AnyBitsClear(data,mask)	( !_AllBitsSet ( data, mask ) )
#define _AllBitsClear(data,mask)	( !_AnyBitSet ( data, mask ) )

/*FHStart
			 Macro:	_Field()

	   Description:	determines the numerical value of the specified bit field

		 Arguments:	<integer type> 'Data' to be tested
					<integer type> bit 'mask' (justified to LSB)
						(mask is same size as data)
					<unsigned integer type> 'shift' position of field LSB
  Global Variables:	none
		   Returns:	field value

	   Constraints:	none
FHEnd*/
#define _Field(data,shift,mask)	( ( (data) >> (shift) ) & (mask) )

/*FHStart
			 Macro:	_SetBit()
					_ClearBit()
					_InvertBit()

	   Description:	sets the specified bit to a particular state

		 Arguments:	<integer type> 'Data' to be modified
					<positive integer> 'bit' number
						(0 <= bit <= bit size of Data)
  Global Variables:	none
		   Returns:	(indirect) specifed bit in 'Data' is forced to the
					specified state; bit 0 is LSB

	   Constraints:	none
FHEnd*/
#define _SetBit(data,bit)		( ( data ) |= ( 1 << ( bit ) ) )
#define _ClearBit(data,bit)    	( ( data ) &= ~( 1 << ( bit ) ) )
#define _InvertBit(data,bit)	( ( data ) ^= ( 1 << ( bit ) ) )

/*FHStart
			 Macro:	_SetBits()
					_ClearBits()
					_InvertBits()

	   Description:	sets the specified bit(s) to a particular state

		 Arguments:	<integer type> 'Data' to be modified
					<integer type> 'bit' mask
						(mask is same size as data)
  Global Variables:	none
		   Returns:	(indirect) bit(s) in 'Data' corresponding to those SET
					(1) in the mask are forced to the specified state

	   Constraints:	none
FHEnd*/
#define _SetBits(data,mask)	   	( ( data ) |= ( mask ) )
#define _ClearBits(data,mask)   ( ( data ) &= ~( mask ) )
#define _InvertBits(data,mask)  ( ( data ) ^= ( mask ) )

/*FHStart
			 Macro:	_SetField()

	   Description:	sets the numerical value of the specified bit field

		 Arguments:	<integer type> 'Data' to be tested
					<integer type> bit 'mask' (justified to LSB)
						(mask is same size as data)
					<unsigned integer type> 'shift' position of field LSB
					<integer type> field 'value'
  Global Variables:	none
		   Returns:	(indirect) field value

	   Constraints:	none
FHEnd*/
#define _SetField(data,mask,shift,value) \
								( ( (data) & ~( (mask) << (shift) ) ) \
									| ( ( (value) & (mask) ) << (shift) ) )

#endif	// _NO

		/*** control flow ***/

/*FHStart
			 Macro:	_RunOnce
					_RunUntilBreak
					_RunForever

	   Description:	prefix for an infinite loop

					use as follows:

						_RunOnce (
							'do stuff here'
							if ( some condition ) {
								break;
							}
							'do more stuff here'
								.
								.
								.
						)

						_RunUntilBreak (
							' do stuff here'
							if ( some condition ) {
								break;
							}
							'do more stuff here'
								.
								.
								.
						)

						_RunForever (
							'do stuff here'
								.
								.
								.
						)

		 Arguments:	none
  Global Variables:	none
		   Returns:	none

	   Constraints:	none
FHEnd*/
#define _RunOnce(code)			do { code } while ( _FALSE )
#define _RunUntilBreak(code)	do { code } while ( _TRUE )
#define _RunForever(code)		do { _RunUntilBreak ( code ); } while ( _TRUE )


		/*** data value test ***/

/*FHStart
			 Macro:	_Negative(val)
					_Zero(val)
					_Positive(val)
					_Sign(val)

	   Description:	tests data values for specified conditions

		 Arguments:	data value
  Global Variables:	none
		   Returns:	condition code

	   Constraints:	none
FHEnd*/
#define _Negative(val)	   		( (val) < _ZERO )
#define _Zero(val)			   	( (val) == _ZERO )
#define _Positive(val)	   		( (val) > _ZERO )


	/*** forward enum references ***/


	/*** forward struct references ***/


	/*** forward type references ***/


	/*** enum definitions ***/


	/*** struct definitions ***/


	/*** type definitions ***/

		/*** generic data types ***/

			/*** generic character types ***/
typedef char					_Char;			/* ASCII character */
typedef _Char *					_PChar;

			/*** integer types ***/
typedef signed char				SInt8;			/* 8-bit signed integer */
typedef SInt8 *					PSInt8;
typedef unsigned char			UInt8; 			/* 8-bit unsigned integer */
typedef UInt8 *					PUInt8;
typedef signed short			SInt16;			/* 16-bit signed integer */
typedef SInt16 *				PSInt16;
typedef unsigned short			UInt16;			/* 16-bit unsigned integer */
typedef UInt16 *				PUInt16;
typedef signed long				SInt32;			/* 32-bit signed integer */
typedef SInt32 *				PSInt32;
typedef unsigned long			UInt32; 		/* 32-bit unsigned integer */
typedef UInt32 *				PUInt32;

#ifdef _WIN32
  typedef __int64				SInt64;			/* 32-bit signed integer */
  typedef SInt64 *				PSInt64;
  typedef unsigned __int64		UInt64;
  typedef UInt64 *				PUInt64;
#else
  #include <stdint.h>
  typedef int64_t				SInt64;			/* 32-bit signed integer */
  typedef SInt64 *				PSInt64;
  typedef uint64_t				UInt64;
  typedef UInt64 *				PUInt64;
#endif

typedef int						SNativeInt;		/* native signed integer */
typedef SNativeInt *			PSNativeInt;
typedef unsigned int			UNativeInt;		/* native unsigned integer */
typedef UNativeInt *			PUNativeInt;

typedef /*pk UInt32*/ unsigned int					FPGAReg;		/* FPGA register bit field */
typedef FPGAReg*				PFPGAReg;

			/*** floating-point types ***/
typedef float					FloatSP;		/* single-precision float */
typedef FloatSP *				PFloatSP;
typedef double					FloatDP;		/* double-precision float */
typedef FloatDP *				PFloatDP;

			/*** complex number types ***/
	/*
		NOTE:	The following construction is specific to Microsoft Visual C++.
				It must be adjusted for other compilers.

				Note that use of complex numbers as an inherent type is only
				supported within C++ programs.
	*/
#ifdef		__cplusplus
typedef std::complex<FloatDP>	ComplexSP;
typedef ComplexSP*				PComplexSP;
typedef std::complex<FloatDP>	ComplexDP;
typedef ComplexDP*				PComplexDP;
#endif		/* __cplusplus */

			/*** null and pointer types ***/
typedef void *	 				_Pointer;		/* generic pointer */
typedef _Pointer *				_PPointer;
typedef _PPointer				_Handle;  		/* generic handle */
typedef _Handle *				_PHandle;

			/*** logical state types ***/
					/*
						NOTE:	For the following I specifically did NOT use the
								standard 'C++' type 'bool' and its corresponding
								constants 'true' and 'false' because I wanted a
								type that would be consistent across 'C' and
								'C++' platforms.  Since '_Bool' is defined in
								the C++ template libraries in the 'std' namespace,
								I also avoided that.
					*/
typedef UInt8					_LBool;	 		/* boolean logic type */
typedef UInt8					_lBool;			/* Also logical boolean - for existing libraries */
typedef _LBool *				_PLBool;
typedef _lBool *				_PlBool;

					/*
						NOTE:	The following were originally defined as _State.  However,
								this definition conflicts with a defnition used in
								the templates in iosfwd which appears when the MS
								extensions are turned off.
					*/
typedef SInt16					_StdState;	 		/* useful states */
typedef _StdState *				_PStdState;
typedef SInt32					_LStdState;
typedef _LStdState *			_PLStdState;

typedef SInt16		   			_Decision;		/* result of a decision */
typedef _Decision *				_PDecision;
typedef SInt32	 				_LDecision;
typedef _LDecision *			_PLDecision;

typedef SInt16		 			_Result;  		/* result of an action */
typedef _Result *  				_PResult;
typedef SInt32	   				_LResult;
typedef _LResult *				_PLResult;
typedef UInt32					_ULResult;
typedef _ULResult *				_PULResult;

#ifdef _WIN32
	#define STDCALL __stdcall
	#if	defined ( HOSTMODULEDLL_EXPORTS )
		#pragma message ( "Exporting 'Host Module' DLL symbols..." )
		#define DLLClass  extern __declspec( dllexport )
	#else		/* NOT HOSTMODULEDLL_EXPORTS */
		#pragma message ( "Importing 'Host Module' DLL symbols..." )
		#define DLLClass	extern _declspec( dllimport )
	#endif		/* HOSTMODULEDLL_EXPORTS */
#else
	#define DLLClass
	#define STDCALL __attribute__((stdcall))
#endif

	/*** bit definitions ***/
/* bit definition */
#define BIT(n)          (1 << n)

#define BIT0            (BIT(0))
#define BIT1            (BIT(1))
#define BIT2            (BIT(2))
#define BIT3            (BIT(3))
#define BIT4            (BIT(4))
#define BIT5            (BIT(5))
#define BIT6            (BIT(6))
#define BIT7            (BIT(7))
#define BIT8            (BIT(8))
#define BIT9            (BIT(9))
#define BIT10           (BIT(10))
#define BIT11           (BIT(11))
#define BIT12           (BIT(12))
#define BIT13           (BIT(13))
#define BIT14           (BIT(14))
#define BIT15           (BIT(15))
#define BIT16           (BIT(16))
#define BIT17           (BIT(17))
#define BIT18           (BIT(18))
#define BIT19           (BIT(19))
#define BIT20           (BIT(20))
#define BIT21           (BIT(21))
#define BIT22           (BIT(22))
#define BIT23           (BIT(23))
#define BIT24           (BIT(24))
#define BIT25           (BIT(25))
#define BIT26           (BIT(26))
#define BIT27           (BIT(27))
#define BIT28           (BIT(28))
#define BIT29           (BIT(29))
#define BIT30           (BIT(30))
#define BIT31           (BIT(31))


	/*** Bit field definitions: instead of using structure bitfield
	     Use the following macros to define bitfield (i.e. for register
	     definitions) ***/

// this type contains bitfield information
// bits  5 -  0 = bitfield size: can encode value from 1 to 64
// bits  6 - 11 = bitfield position: can encode value from 0 to 63.
//				 Give the position of the LSB of the mask.
// bits 12 - 14 = size in byte of the register where this bitfield is defined in.
//				  0b00 = 1 byte; 0b01 = 2 bytes; 0b10 = 4 bytes; 0b11 = 8 bytes.
// bits: 15  -   8 7   -   8
//       00rr pppp ppss ssss
// p represent position bits; s represent size bits; r represent register size bits.
typedef UInt16	_BITFIELD;

// Mask to access the position information inside _BITFIELD
#define MASK_BIT_POS 0x0fc0
// Mask to access the size information inside _BITFIELD
#define MASK_BIT_SIZE 0x003f
// Mask to access the register size information insde _BITFIELD
#define MASK_REG_SIZE 0x3000

// Define bit field with position and size. For Position 0 = LSB.
#define _DEF_BITFIELD(bitPos, size)			(((_BITFIELD)bitPos) << 6 | ((_BITFIELD)size))
// Get bit field size from _BITFIELD
#define GET_SIZE_BITFIELD(bitField)			(((_BITFIELD)bitField) & MASK_BIT_SIZE)
// Get bit field position from _BITFIELD
#define GET_POS_BITFIELD(bitField)			((((_BITFIELD)bitField) & MASK_BIT_POS) >> 6)
// Create a mask for the bitfield define by its position and size (only used in setBitFieldXX)
#define CREATE_BIT_MASK(pos, size)			(~((( 1 << ((_BITFIELD)size) ) - 1) << (_BITFIELD)pos))
// Get the value of the bit field coded in 32bits (only used in setBitFieldXX)
#define GET_VALBITFIELD(val, bitField)		(((_BITFIELD)val) << (GET_POS_BITFIELD(bitField)))

// set the bitfield value of a 32bit register
#if 0
	inline void setBitField32(UInt32* p, _BITFIELD bitField, UInt32 uiValue)
	{
		// Add encoding of register size inside _BITFIELD
		*p &= CREATE_BIT_MASK(GET_POS_BITFIELD(bitField), GET_SIZE_BITFIELD(bitField));
		*p |= GET_VALBITFIELD(uiValue, bitField);
	};
#else

#define setBitField32(p,b,v)				(*((PUInt32)(p)) =					\
												(*((PUInt32)(p)) &				\
													CREATE_BIT_MASK(			\
														GET_POS_BITFIELD(b),	\
														GET_SIZE_BITFIELD(b)))	\
													| GET_VALBITFIELD((v),(b)))
#endif

// get the bitfield value
#if 0
	inline UInt32 getBitField32(UInt32 p, _BITFIELD bitField)
	{
		p &= ~CREATE_BIT_MASK(GET_POS_BITFIELD(bitField), GET_SIZE_BITFIELD(bitField));
		return (p >> GET_POS_BITFIELD(bitField));
	};
#else
#define getBitField32(p,b)					((((UInt32)(p)) &					\
													(~CREATE_BIT_MASK(			\
														GET_POS_BITFIELD(b),	\
														GET_SIZE_BITFIELD(b))))	\
													>> GET_POS_BITFIELD(b))


#endif

/* The following is used for vaarg processing differences between windows and QNX when routines
 * are called with nested va_args calls.
 */
#ifdef _WIN32
	#define VARG_START_EXT( valistNew, valistArg, vaFormat )	va_start( valistNew, vaFormat )
#else
	#define VARG_START_EXT( valistNew, valistArg, vaFormat )	va_copy( valistNew, *valistArg )
#endif

	/*** external data references ***/


	/*** external function prototypes ***/

#endif		/* stdhdr_h */

