/*//////////////////////////////////////////////////////////////////////////////
//MDStart
//	(C) Copyright 2008 Triple Ring Technologies, Incorporated
//	All Rights Reserved
//
//	THIS IS THE UNPUBLISHED PROPRIETARY SOURCE CODE OF TRIPLE RING TECHNOLOGIES.
//	The copyright notice above does not evidence any actual or intended
//	publication of such source code.
//
//
//! Description:	Result code declarations for the standard TRT libraries.
//!
//!		Result codes are defined to be 32-bit values comprised of 6 bit fields.
//!		The bit fields are defined as follows:
//!
//!   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1						\n
//!   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0	\n
//!  +---+-+-+---------------+---------------+-----------------------+	\n
//!  |Sev|C|R|   SubSys  ID  |  Module ID    |        Code           |	\n
//!  +---+-+-+---------------+---------------+-----------------------+	\n
//!
//!  where:\n
//!
//!		Sev - is the severity code (2-bits)								\n
//!
//!			00 - Informational											\n
//!			01 - Warning												\n
//!			10 – Error													\n
//!			11 – Panic (Used for software crash)						\n
//!
//!		C - is the Customer code flag (1-bit)							\n
//!
//!		R - is a reserved bit (don't use it)							\n
//!
//!		Subsystem ID – This is a project-specific enumerated identification
//!			 code. 
//!
//!     Code - is the facility's status code (Unique error) (12-bits) (0-4095)
//!
////MDEnd
//////////////////////////////////////////////////////////////////////////////*/

/*IDStart*/
#ifndef		StdError_h
#define		StdError_h
/*IDEnd*/


#include "stdhdr.h"

	// Declarations for field locations and masks

#define LRESULT_FLD_SEVERITY_SHIFT	30
#define LRESULT_FLD_SEVERITY_MASK	0x03
#define LRESULT_FLD_SUBSYSTEM_SHIFT	20
#define LRESULT_FLD_SUBSYSTEM_MASK	0xFF
#define LRESULT_FLD_MODULE_SHIFT	12
#define LRESULT_FLD_MODULE_MASK		0xFF
#define LRESULT_FLD_ERRORCODE_SHIFT	0
#define LRESULT_FLD_ERRORCODE_MASK	0xFFF


	//! The LRESULT_FIELD union may be used to programmatically 
	//! assemble and disassemble error codes.

		// Disable Microsoft warning C4214. With ANSI compatibility, bit fields can only
		// be of int type. This can cause problems on other platforms.
#if defined ( WIN32 )
	#pragma warning ( disable : 4214 )
#endif

typedef union {
	_LResult	bits ;
	struct {
		UInt32	Code			: 12 ;
		UInt32	Module			: 8 ;
		UInt32	Subsystem		: 8 ;
		UInt32	Reserved		: 1 ;
		UInt32	Customer		: 1 ;
		UInt32	Severity		: 2 ;
	} e ;

} LRESULT_FIELD ;

#if defined ( WIN32 )
	#pragma warning ( default : 4214 )
#endif



		// Error Serverity
enum LRESULT_SEVERITY {

	LRESULT_SEVERITY_INFO		= 0,
	LRESULT_SEVERITY_WARNING	= 1,
	LRESULT_SEVERITY_ERROR		= 2,
	LRESULT_SEVERITY_PANIC		= 3
};


	/*** Access macros ***/

	//! Macro MAKE_LRESULT() may be used to build result/error values.
	//! 
	//! The macro has for values:
	//!
	//! \param v	LRESULT severity code may be one of the values 
	//!					listed in the enumeration LRESULT_SEVERITY.

#define MAKE_LRESULT(v,b,m,c)		(_LResult)(\
										(((v)&LRESULT_FLD_SEVERITY_MASK)		  \
												<<LRESULT_FLD_SEVERITY_SHIFT)	| \
										(((b)&LRESULT_FLD_SUBSYSTEM_MASK)		  \
												<<LRESULT_FLD_SUBSYSTEM_SHIFT)	| \
										(((m)&LRESULT_FLD_MODULE_MASK)			  \
												<<LRESULT_FLD_MODULE_SHIFT)		| \
										(((c)&LRESULT_FLD_ERRORCODE_MASK)		  \
												<<LRESULT_FLD_ERRORCODE_SHIFT))

		// Retrieving component result fields
#define RESULT_SEVERITY(l)			_Field(l,LRESULT_FLD_SEVERITY_MASK,\
										LRESULT_FLD_SEVERITY_SHIFT)
#define RESULT_SUBSYSTEM(l)			_Field(l,LRESULT_FLD_SUBSYSTEM_SHIFT,\
										LRESULT_FLD_SUBSYSTEM_MASK)
#define RESULT_MODULE(l)			_Field(l,LRESULT_FLD_MODULE_SHIFT,\
										LRESULT_FLD_MODULE_MASK)
#define RESULT_CODE(l)				_Field(l,LRESULT_FLD_ERRORCODE_SHIFT,\
										LRESULT_FLD_ERRORCODE_MASK)

		// Determining result code (error) severity
#define LRESULT_IS_INFO(l)			(LRESULT_SEVERITY_INFO==LRESULT_SEVERITY(l))
#define LRESULT_IS_WARNING(l)		(LRESULT_SEVERITY_WARNING==LRESULT_SEVERITY(l))
#define LRESULT_IS_ERROR(l)			(LRESULT_SEVERITY_ERROR==LRESULT_SEVERITY(l))
#define LRESULT_IS_PANIC(l)			(LRESULT_SEVERITY_PANIC==LRESULT_SEVERITY(l))

		// Enumeration for subsystem ID codes; only generic subsystem 
		// identifiers should be specified here

enum LRESULT_SUBSYSTEM_ID {

	LRESULT_SUBID_UNSPECIFIED		= 0,
	LRESULT_SUBID_OSIF,						// Operating system interface layer
	LRESULT_SUBID_STDLIB,					// Standard TRT library code
	LRESULT_SUBID_EVLOG,					// eventlog handler
	LRESULT_SUBID_EMCP,						// Emcp data
	LRESULT_SUBID_SNMP,						// SNMP protocol
	LRESULT_SUBID_FPGA,						// FPGA device
	LRESULT_SUBID_CLIENT_BASE		= 20	// Starting value for application 
											//		code
} ;


		/*** error codes ***/
		/**/

/* error severity codes */
#define LRESULT_NO_ERROR			0
#define LRESULT_OK					0


	/* some useful error codes */
#define MAKE_GENERIC_LRESULT(n)		MAKE_LRESULT(LRESULT_SEVERITY_ERROR,	\
										LRESULT_SUBID_UNSPECIFIED,0,(n))

#define LRESULT_GEN_FAILURE			MAKE_GENERIC_LRESULT(1)	
#define LRESULT_NOT_ENOUGH_MEMORY	MAKE_GENERIC_LRESULT(2)
#define LRESULT_INVALID_HANDLE		MAKE_GENERIC_LRESULT(3)
#define LRESULT_INVALID_FUNCTION	MAKE_GENERIC_LRESULT(4)
#define LRESULT_INVALID_PARAMETER	MAKE_GENERIC_LRESULT(5)
#define LRESULT_INVALID_CONTEXT		MAKE_GENERIC_LRESULT(6)
#define LRESULT_INVALID_DATA		MAKE_GENERIC_LRESULT(7)
#define LRESULT_INVALID_DEVICE		MAKE_GENERIC_LRESULT(8)
#define LRESULT_DEVICE_NOT_READY	MAKE_GENERIC_LRESULT(9)
#define LRESULT_DEVICE_BUSY			MAKE_GENERIC_LRESULT(10)
#define LRESULT_ACCESS_DENIED		MAKE_GENERIC_LRESULT(11)
#define LRESULT_FILE_NOT_FOUND		MAKE_GENERIC_LRESULT(12)
#define LRESULT_WRITE				MAKE_GENERIC_LRESULT(13)
#define LRESULT_READ				MAKE_GENERIC_LRESULT(14)
#define LRESULT_EOF					MAKE_GENERIC_LRESULT(15)
#define LRESULT_TIMEOUT				MAKE_GENERIC_LRESULT(16)
#define LRESULT_OVERFLOW			MAKE_GENERIC_LRESULT(17)
#define LRESULT_UNDERFLOW			MAKE_GENERIC_LRESULT(18)

	/* start application-defined error codes here */
#define LRESULT_APPLICATION			100


	/* some useful information codes */
#define LRESULT_INFO_CONTINUE		MAKE_LRESULT(LRESULT_SEVERITY_INFO,	\
										LRESULT_SUBID_UNSPECIFIED,0,1)


#endif /* StdError_h */
