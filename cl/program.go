package cl

// #include <stdlib.h>
// #include <OpenCL/opencl.h>
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

type BuildError string

func (e BuildError) Error() string {
	return fmt.Sprintf("cl: build error (%s)", string(e))
}

type Program struct {
	clProgram C.cl_program
	devices   []*Device
}

func releaseProgram(p *Program) {
	if p.clProgram != nil {
		C.clReleaseProgram(p.clProgram)
		p.clProgram = nil
	}
}

func (p *Program) Release() {
	releaseProgram(p)
}

func (p *Program) BuildProgram(devices []*Device, options string) error {
	if err := C.clBuildProgram(p.clProgram, 0, nil, nil, nil, nil); err != C.CL_SUCCESS {
		buffer := make([]byte, 4096)
		var bLen C.size_t
		var err C.cl_int
		for i := 2; i >= 0; i-- {
			err = C.clGetProgramBuildInfo(p.clProgram, p.devices[0].id, C.CL_PROGRAM_BUILD_LOG, C.size_t(len(buffer)), unsafe.Pointer(&buffer[0]), &bLen)
			if err == C.CL_INVALID_VALUE && i > 0 && bLen < 1024*1024 {
				// INVALID_VALUE probably means our buffer isn't large enough
				buffer = make([]byte, bLen)
			} else {
				break
			}
		}
		if err != C.CL_SUCCESS {
			return toError(err)
		}
		return BuildError(string(buffer[:bLen]))
	}
	return nil
}

func (p *Program) CreateKernel(name string) (*Kernel, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var err C.cl_int
	clKernel := C.clCreateKernel(p.clProgram, cName, &err)
	if err != C.CL_SUCCESS {
		return nil, toError(err)
	}
	kernel := &Kernel{clKernel: clKernel, name: name}
	runtime.SetFinalizer(kernel, releaseKernel)
	return kernel, nil
}
