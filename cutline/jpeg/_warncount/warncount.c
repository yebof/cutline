/* warncount: the libjpeg zero-warning check, run against the PINNED reference decoder, i.e. the libjpeg-turbo 2.1.0
 * build inside the jpeglib Python package (cjpeglib_turbo210.abi3.so), loaded with dlopen. Two decodes per file:
 * (0) jpeg_read_header + jpeg_read_coefficients, (1) a full decompress with library defaults (JDCT_ISLOW, fancy
 * upsampling, JCS_RGB). A custom error manager: error_exit longjmps; warnings are counted by libjpeg's own
 * emit_message into err->num_warnings (msg_level -1); output_message is silenced. The helper must be compiled for the
 * build's JPEG_LIB_VERSION (-DLIBVER; the struct layout differs between 62, 70 and 80).
 *
 * Build (cutline.jpeg.libjpeg does this on first use):
 *   gcc -O2 -shared -fPIC -I. -DLIBVER=70 -o warncount70.so warncount.c -ldl
 * run(): returns 0 (no error) or 1 (error exit); fills the warning count and the first message.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <dlfcn.h>
#ifndef LIBVER
#define LIBVER 62
#endif
#include "jpeglib.h"
#ifndef MAX_MEMORY_TO_USE
#define MAX_MEMORY_TO_USE (512L * 1024 * 1024)
#endif

typedef struct jpeg_error_mgr *(*f_std_error)(struct jpeg_error_mgr *);
typedef void (*f_create)(j_decompress_ptr, int, size_t);
typedef void (*f_destroy)(j_decompress_ptr);
typedef void (*f_memsrc)(j_decompress_ptr, const unsigned char *, unsigned long);
typedef int (*f_read_header)(j_decompress_ptr, boolean);
typedef jvirt_barray_ptr *(*f_read_coefs)(j_decompress_ptr);
typedef boolean (*f_start)(j_decompress_ptr);
typedef JDIMENSION (*f_scanlines)(j_decompress_ptr, JSAMPARRAY, JDIMENSION);
typedef boolean (*f_finish)(j_decompress_ptr);

static f_std_error p_std_error; static f_create p_create; static f_destroy p_destroy; static f_memsrc p_memsrc;
static f_read_header p_read_header; static f_read_coefs p_read_coefs; static f_start p_start;
static f_scanlines p_scanlines; static f_finish p_finish;

struct my_err { struct jpeg_error_mgr pub; jmp_buf jb; char msg[JMSG_LENGTH_MAX]; int have_msg; };

static void my_error_exit(j_common_ptr cinfo) {
  struct my_err *e = (struct my_err *)cinfo->err;
  if (!e->have_msg) { (*cinfo->err->format_message)(cinfo, e->msg); e->have_msg = 1; }
  longjmp(e->jb, 1);
}
static void my_output_message(j_common_ptr cinfo) {  /* called by emit_message for the first warning */
  struct my_err *e = (struct my_err *)cinfo->err;
  if (!e->have_msg) { (*cinfo->err->format_message)(cinfo, e->msg); e->have_msg = 1; }
}

int load(const char *so) {
  void *h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
  if (!h) return -1;
  p_std_error = (f_std_error)dlsym(h, "jpeg_std_error");
  p_create = (f_create)dlsym(h, "jpeg_CreateDecompress");
  p_destroy = (f_destroy)dlsym(h, "jpeg_destroy_decompress");
  p_memsrc = (f_memsrc)dlsym(h, "jpeg_mem_src");
  if (!p_memsrc) p_memsrc = (f_memsrc)dlsym(h, "jpeg_mem_src_tj");
  p_read_header = (f_read_header)dlsym(h, "jpeg_read_header");
  p_read_coefs = (f_read_coefs)dlsym(h, "jpeg_read_coefficients");
  p_start = (f_start)dlsym(h, "jpeg_start_decompress");
  p_scanlines = (f_scanlines)dlsym(h, "jpeg_read_scanlines");
  p_finish = (f_finish)dlsym(h, "jpeg_finish_decompress");
  return (p_std_error && p_create && p_destroy && p_memsrc && p_read_header && p_read_coefs && p_start && p_scanlines
          && p_finish) ? 0 : -2;
}

/* mode 0: coefficients; mode 1: full RGB decompress. Returns 0 on no error, 1 on error exit; *nwarn = warnings. */
int run(const unsigned char *buf, unsigned long len, int mode, long *nwarn, char *msg, int msglen) {
  struct jpeg_decompress_struct cinfo;
  struct my_err err;
  volatile int rc = 0;
  unsigned char *volatile row = NULL;   /* volatile: assigned after setjmp, read after longjmp (C11 7.13.2.1) */
  memset(&cinfo, 0, sizeof(cinfo));
  memset(&err, 0, sizeof(err));
  cinfo.err = p_std_error(&err.pub);
  err.pub.error_exit = my_error_exit;
  err.pub.output_message = my_output_message;
  if (setjmp(err.jb)) { rc = 1; goto done; }
  p_create(&cinfo, JPEG_LIB_VERSION, sizeof(cinfo));
  /* Defence in depth for the container size bound: cap libjpeg's virtual-array memory. The pinned build has no
   * backing store, so a file needing more fails with an error exit (never reached by a file inside the size bound). */
  cinfo.mem->max_memory_to_use = MAX_MEMORY_TO_USE;
  p_memsrc(&cinfo, buf, len);
  p_read_header(&cinfo, TRUE);
  if (mode == 0) {
    p_read_coefs(&cinfo);
  } else {
    cinfo.out_color_space = JCS_RGB;
    cinfo.dct_method = JDCT_ISLOW;
    cinfo.do_fancy_upsampling = TRUE;
    p_start(&cinfo);
    row = (unsigned char *)malloc((size_t)cinfo.output_width * cinfo.output_components);
    while (cinfo.output_scanline < cinfo.output_height) { JSAMPROW r = row; p_scanlines(&cinfo, &r, 1); }
  }
  p_finish(&cinfo);
done:
  *nwarn = err.pub.num_warnings;
  if (msg && msglen > 0) { strncpy(msg, err.msg, msglen - 1); msg[msglen - 1] = 0; }
  free(row);
  p_destroy(&cinfo);
  return rc;
}

int libver(void) { return JPEG_LIB_VERSION; }
int structsize(void) { return (int)sizeof(struct jpeg_decompress_struct); }
