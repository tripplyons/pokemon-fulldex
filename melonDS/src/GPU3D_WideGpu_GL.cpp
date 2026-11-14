// ssrecomp: the widescreen view's polygons drawn by OpenGL (GPU3D_WideGpu.h): on a PC, to test
// the graphics chip's version of the picture against GPU3D_Wide.cpp's own rasteriser, and for
// WORLD ON GPU in the PC all-games program (high WORLD DETAIL, 4K, without the processor's cost).
// Its context is made by the frontend's hook (GLThreadRenderer::CreateContext). In the headless
// tests it is made on the thread that draws and stays current there (everything runs on one
// thread). In play (WideGpuGLThreaded) it lives on a thread of its own (WideGpuGLWorker), which
// draws each picture and copies it out while the emulation goes on: the frames are drawn by the
// 3D render thread and waited for on the main one, where SDL may draw with OpenGL too, and a
// context can be current on one thread at a time.
#ifdef OGLRENDERER_ENABLED
#include <chrono>
#include "GPU3D_WideGpu.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include "OpenGLSupport.h"
#ifdef __ANDROID__
#include <sched.h>
#include <unistd.h>
#endif
#include "GPU3D_GLThread.h"

#include "wide_gpu/wide_vsh_text.h"
#include "wide_gpu/wide_fsh_text.h"
#include "wide_gpu/final_vsh_text.h"
#include "wide_gpu/final_fsh_text.h"

namespace melonDS
{
void* (*WideGpuGLGetProc)(const char* name) = nullptr;
bool WideGpuGLThreaded = false;
void (*WideGpuGLRelease)() = nullptr;
#ifdef __ANDROID__
bool (*WideGpuGLCreateShared)(std::string& info) = nullptr;
std::atomic<bool> WideZcWantPixels { false };

namespace
{
// SSR_WIDE_ZEROCOPY (GPU3D_WideGpu.h): each slot's picture and where it is, shared by the
// worker's thread and the frontend's (under ZcMutex).
struct ZcSlot
{
    GLuint Offered = 0;          // the frontend's texture for the slot's next pictures
    int OfferedW = 0, OfferedH = 0;
    GLsync ReadFence = nullptr;  // after the frontend's last draws from the slot's texture
    u32* Pixels = nullptr;       // the slot's memory (Target)
    int W = 0, H = 0;
    GLuint Drawn = 0;            // the texture the slot's picture is in (0: only in the memory)
    GLuint Drawing = 0;          // the texture the worker is drawing into now
    bool HasPixels = true;       // the memory holds the picture too
};
std::mutex ZcMutex;
ZcSlot Zc[3];
bool ZcOn = false; // a worker draws into the frontend's textures

void ZcRelease(bool inContext)
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    for (ZcSlot& z : Zc)
    {
        if (z.ReadFence && inContext) glDeleteSync(z.ReadFence);
        z = ZcSlot();
    }
    ZcOn = false;
}
}
#endif

namespace
{
typedef void (*ClipControlFn)(GLenum origin, GLenum depth);
typedef void (*InvalidateFn)(GLenum target, GLsizei count, const GLenum* attachments);
constexpr GLenum LowerLeft = 0x8CA1, ZeroToOne = 0x935F; // GL_LOWER_LEFT, GL_ZERO_TO_ONE

// OpenGL ES (a phone: Android), where a few things are spelled differently: see Source, Texture,
// Fetch and the final pass. Set by Init from the context's version.
bool IsES = false;
// What OpenGL ES needs done differently and desktop OpenGL can do too (vertices as attributes,
// no noperspective, red and blue swapped for the read-back): on with OpenGL ES, and on a PC with
// SSR_WIDEGPU_ES=1, so that the headless tests can compare those paths' pictures with the others.
bool EsPaths = false;

// The shaders are written for deko3D's compiler (GLSL 4.60); OpenGL 4.3 takes them as 4.30,
// OpenGL ES 3.2 as 3.20 es with its precisions named and WIDE_ES defined (wide_vsh.glsl).
std::string Source(const unsigned char* text, size_t size)
{
    std::string s(reinterpret_cast<const char*>(text), size);
    const size_t at = s.find("#version 460");
    if (at != std::string::npos)
        s.replace(at, 12, IsES ? "#version 320 es\n#define WIDE_ES 1\nprecision highp float;\nprecision highp int;\n"
                                 "precision highp sampler2D;\nprecision highp usampler2D;\nprecision highp usamplerBuffer;"
                               : EsPaths ? "#version 430 core\n#define WIDE_ES 1" : "#version 430 core");
    return s;
}

// The same with a #define after the #version line (a variant: see wide_fsh.glsl).
std::string Variant(std::string s, const char* define)
{
    const size_t eol = s.find('\n');
    s.insert(eol + 1, std::string("#define ") + define + " 1\n");
    return s;
}

class WideGpuGL : public WideGpuBackend
{
public:
    const char* Name() const override { return "OpenGL"; }

    bool Init(std::string& info)
    {
        // (SSR_WIDEGPU_SHADER=texbuf: the Switch's variant, the texture memory read as buffer
        // textures, to check its pictures here.)
        const char* glVersion = (const char*)glGetString(GL_VERSION);
        IsES = glVersion && strstr(glVersion, "OpenGL ES");
        EsPaths = IsES || getenv("SSR_WIDEGPU_ES") != nullptr;
        if (EsPaths && !IsES) fprintf(stderr, "[widegpu] OpenGL: the OpenGL ES paths (SSR_WIDEGPU_ES)\n");
        if (IsES)
        {
            // (What the shaders need of it: storage buffers in both stages.)
            GLint vs = 0, fs = 0;
            glGetIntegerv(0x90D6 /* GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS */, &vs);
            glGetIntegerv(0x90DA /* GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS */, &fs);
            fprintf(stderr, "[widegpu] %s: storage buffers a vertex shader %d, a fragment shader %d\n", glVersion, (int)vs, (int)fs);
            // (The vertex shader takes its vertices as attributes there: phones' graphics chips
            // give it none, Mali 0.)
            if (fs < 3)
            {
                info = "this graphics chip gives the shaders too few storage buffers";
                return false;
            }
        }
        const char* variant = getenv("SSR_WIDEGPU_SHADER");
        // OpenGL ES (a phone): the buffer-texture shader and its decoded textures by default,
        // 9.2 -> 8.4 ms a 720-line picture on an Adreno, the same pixels (SSR_WIDEGPU_SHADER=ssbo:
        // the storage-buffer one).
        TexBuf = variant ? !strcmp(variant, "texbuf") : IsES;
        std::string fsh = Source(wide_fsh_text, sizeof(wide_fsh_text));
        if (TexBuf)
        {
            fsh = Variant(fsh, "WIDE_TEXBUF");
            fprintf(stderr, "[widegpu] OpenGL: the buffer-texture fragment shader\n");
        }
        // (SSR_WIDEGPU_ATTR8=1: the depth and fog flag in an RGB10_A2UI target, 4 bytes a pixel,
        // instead of RG32UI, 8: less tile memory a pixel on a tiled graphics chip, and half the
        // bytes written out and read back by the final pass. The same values: wide_fsh.glsl.)
        // On by default with OpenGL ES (a phone); SSR_WIDEGPU_ATTR8=0 or 1 chooses.
        static const char* attr8Env = getenv("SSR_WIDEGPU_ATTR8");
        AttrPacked = attr8Env ? attr8Env[0] == '1' : IsES;
        if (AttrPacked)
        {
            fsh = Variant(fsh, "WIDE_ATTR_PACKED");
            fprintf(stderr, "[widegpu] OpenGL: depth and fog in RGB10_A2UI (SSR_WIDEGPU_ATTR8)\n");
        }
        std::string finalFsh = Source(final_fsh_text, sizeof(final_fsh_text));
        if (EsPaths) finalFsh = Variant(finalFsh, "WIDE_SWAP_RB");
        if (AttrPacked) finalFsh = Variant(finalFsh, "WIDE_ATTR_PACKED");
        if (!OpenGL::CompileVertexFragmentProgram(MainProgram, Source(wide_vsh_text, sizeof(wide_vsh_text)),
                fsh, "WideMain", {}, {}) ||
            !OpenGL::CompileVertexFragmentProgram(EarlyProgram, Source(wide_vsh_text, sizeof(wide_vsh_text)),
                Variant(fsh, "WIDE_EARLY"), "WideEarly", {}, {}) ||
            !OpenGL::CompileVertexFragmentProgram(FinalProgram, Source(final_vsh_text, sizeof(final_vsh_text)),
                finalFsh, "WideFinal", {}, {}))
        {
            info = "the shaders did not compile (see the log)";
            return false;
        }
        if (EsPaths)
        {
            // (What each draw is told, as a plain uniform there: see wide_vsh.glsl.)
            DrawLoc[0] = glGetUniformLocation(MainProgram, "uDraw");
            DrawLoc[1] = glGetUniformLocation(EarlyProgram, "uDraw");
        }
        glGenBuffers(1, &FrameUBO);
        glGenBuffers(1, &DrawUBO);
        glGenBuffers(4, SSBO);
        glGenTextures(1, &PolyTex);
        glGenBuffers(1, &ReadPBO);
#ifdef __ANDROID__
        if (ZeroCopy)
        {
            glGenFramebuffers(1, &ZcFBO);
            glGenBuffers(1, &EdgePBO);
        }
#endif
        glGenVertexArrays(1, &EmptyVAO);
        if (EsPaths)
        {
            // The vertices as attributes, from the buffer the desktop's shader reads as storage.
            glGenVertexArrays(1, &VertexVAO);
            glBindVertexArray(VertexVAO);
            glBindBuffer(GL_ARRAY_BUFFER, SSBO[0]);
            const GLsizei stride = (GLsizei)sizeof(WideGpuVertex);
            for (GLuint i = 0; i < 4; i++) glEnableVertexAttribArray(i);
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (const void*)0);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (const void*)16);
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, (const void*)32);
            glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, stride, (const void*)36);
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 512 * 1024, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[3]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 128 * 1024, nullptr, GL_DYNAMIC_DRAW);
        glGenTextures(4, BufTex);
        glGenBuffers(1, &AtlasBuf);
        glBindBuffer(GL_TEXTURE_BUFFER, AtlasBuf);
        glBufferData(GL_TEXTURE_BUFFER, WideGpuAtlas::Capacity * 4, nullptr, GL_DYNAMIC_DRAW);
        const GLenum formats[4] = { GL_R8UI, GL_R16UI, GL_R16UI, GL_R32UI };
        for (int i = 0; i < 4 && (TexBuf || !IsES); i++)
        {
            glBindTexture(GL_TEXTURE_BUFFER, BufTex[i]);
            glTexBuffer(GL_TEXTURE_BUFFER, formats[i], i == 3 ? AtlasBuf : SSBO[i < 2 ? 2 : 3]);
        }
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_UNIFORM_BUFFER, FrameUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(WideGpuFrameConstants), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, DrawUBO);
        glBufferData(GL_UNIFORM_BUFFER, 16, nullptr, GL_DYNAMIC_DRAW);
        if (WideGpuGLGetProc) ClipControl = (ClipControlFn)WideGpuGLGetProc(IsES ? "glClipControlEXT" : "glClipControl");
        // (OpenGL ES 3.0 and OpenGL 4.3; glad loads it for neither on OpenGL ES.)
        if (WideGpuGLGetProc) Invalidate = (InvalidateFn)WideGpuGLGetProc("glInvalidateFramebuffer");
        if (IsES)
        {
            // (Only where the extension is really there: a name alone may be answered with a stub.)
            const char* ext = (const char*)glGetString(GL_EXTENSIONS);
            if (!ext || !strstr(ext, "GL_EXT_clip_control")) ClipControl = nullptr;
            fprintf(stderr, "[widegpu] depth from 0 to 1 (glClipControl): %s\n", ClipControl ? "yes" : "no");
        }
        return glGetError() == GL_NO_ERROR;
    }

    ~WideGpuGL() override
    {
        glDeleteProgram(MainProgram);
        glDeleteProgram(EarlyProgram);
        glDeleteProgram(FinalProgram);
        glDeleteBuffers(1, &FrameUBO);
        glDeleteBuffers(1, &DrawUBO);
        glDeleteTextures(4, BufTex);
        glDeleteTextures(1, &PolyTex);
        glDeleteBuffers(1, &AtlasBuf);
        glDeleteBuffers(4, SSBO);
        glDeleteBuffers(1, &ReadPBO);
#ifdef __ANDROID__
        if (ZcFBO) glDeleteFramebuffers(1, &ZcFBO);
        if (EdgePBO) glDeleteBuffers(1, &EdgePBO);
        if (WriteFence) glDeleteSync(WriteFence);
#endif
        glDeleteVertexArrays(1, &EmptyVAO);
        if (VertexVAO) glDeleteVertexArrays(1, &VertexVAO);
        FreeTargets();
    }

    // (Drawn at once, so always finished; SSR_WIDE_GL_LATE=1 says it is not, to test what
    // GPU3D_Wide.cpp does with a picture the graphics chip is late with.)
    bool TakesAtlas() const override
    {
        static const bool off = getenv("SSR_WIDEGPU_ATLAS") && !strcmp(getenv("SSR_WIDEGPU_ATLAS"), "0");
        return TexBuf && !off;
    }

    bool Done(int /*slot*/) override
    {
        static const bool late = getenv("SSR_WIDE_GL_LATE") != nullptr;
        return !late;
    }

    u32* Target(int slot, int w, int h) override
    {
        Slots[slot].resize((size_t)w * h);
#ifdef __ANDROID__
        if (ZeroCopy)
        {
            std::lock_guard<std::mutex> lock(ZcMutex);
            Zc[slot].Pixels = Slots[slot].data();
            Zc[slot].W = w;
            Zc[slot].H = h;
        }
#endif
        return Slots[slot].data();
    }

    bool Submit(WideGpuFrame& f, int slot) override
    {
        if (f.W != W || f.H != H) MakeTargets(f.W, f.H);
        // Rows from the bottom of the view are the first in memory (see wide_vsh.glsl).
        f.Constants.Scale[2] = -1.0f;
        // Depth from 0 to 1 where glClipControl is there (OpenGL 4.5), otherwise -1 to 1 (which
        // loses W-buffering's 1/W: its values are tiny).
        f.Constants.Scale[3] = ClipControl ? 0.0f : 1.0f;
        if (ClipControl) ClipControl(LowerLeft, ZeroToOne);
        // (SSR_WIDEGPU_NOTEX=1: no textures, only to measure what they cost.)
        static const bool noTex = getenv("SSR_WIDEGPU_NOTEX") != nullptr;
        if (noTex) f.Constants.Fog[3] |= 4;

        glBindBuffer(GL_UNIFORM_BUFFER, FrameUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(f.Constants), &f.Constants);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, FrameUBO);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, DrawUBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[0]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max<size_t>(16, f.Vertices.size() * sizeof(WideGpuVertex)), f.Vertices.data(), GL_STREAM_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max<size_t>(16, f.Polygons.size() * sizeof(WideGpuPolygon)), f.Polygons.data(), GL_STREAM_DRAW);
        if (EsPaths)
        {
            // The polygons for wide_vsh.glsl's buffer texture (glBufferData gave the buffer new
            // storage, so the texture is pointed at it again).
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, PolyTex);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, SSBO[1]);
        }
        if (!f.TexVersion || f.TexVersion != TexVersion)
        {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[2]);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 512 * 1024, f.TexMem);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[3]);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 128 * 1024, f.PalMem);
            TexVersion = f.TexVersion;
        }
        for (int i = 0; i < 4; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, SSBO[i]);
        if (f.AtlasOn)
        {
            const WideGpuAtlas& a = f.Atlas;
            glBindBuffer(GL_TEXTURE_BUFFER, AtlasBuf);
            if (!AtlasSynced)
            {
                glBufferSubData(GL_TEXTURE_BUFFER, 0, (GLsizeiptr)a.Used * 4, a.Texels.data());
                AtlasSynced = true;
            }
            else
                for (const auto& [first, n] : a.Uploads)
                    glBufferSubData(GL_TEXTURE_BUFFER, (GLintptr)first * 4, (GLsizeiptr)n * 4, a.Texels.data() + first);
        }
        if (TexBuf)
            for (int i = 0; i < 4; i++)
            {
                glActiveTexture(GL_TEXTURE0 + i);
                glBindTexture(GL_TEXTURE_BUFFER, BufTex[i]);
            }
        glActiveTexture(GL_TEXTURE0);

        glBindFramebuffer(GL_FRAMEBUFFER, MainFBO);
        const GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, bufs);
        glViewport(0, 0, W, H);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glDisablei(GL_BLEND, 0);
        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glStencilMask(0xFF);
        glClearBufferfv(GL_COLOR, 0, f.ClearColor);
        const GLuint attr[4] = { AttrPacked ? (f.ClearDepth & 0x3FF) : f.ClearDepth,
                                 AttrPacked ? ((f.ClearDepth >> 10) & 0x3FF) : f.ClearFog,
                                 AttrPacked ? ((f.ClearDepth >> 20) & 0x3FF) : 0u,
                                 AttrPacked ? f.ClearFog : 0u };
        glClearBufferuiv(GL_COLOR, 1, attr);
        glClearBufferfi(GL_DEPTH_STENCIL, 0, f.ClearDepthValue, f.ClearStencil);

        glUseProgram(MainProgram);
        glBindVertexArray(EsPaths ? VertexVAO : EmptyVAO);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
        bool earlyBound = false;
        static const bool noDraw = getenv("SSR_WIDEGPU_NODRAW") != nullptr; // (test: the cost without the polygons)
        for (const WideGpuDraw& d : f.Draws)
        {
            if (noDraw) break;
            if (d.Early != earlyBound)
            {
                glUseProgram(d.Early ? EarlyProgram : MainProgram);
                earlyBound = d.Early;
            }
            const u32 params[4] = { d.First, d.Pixels, (u32)d.DepthBias, f.WDepth ? 1u : 0u };
            if (EsPaths) glUniform4ui(DrawLoc[earlyBound ? 1 : 0], params[0], params[1], params[2], params[3]);
            else
            {
                glBindBuffer(GL_UNIFORM_BUFFER, DrawUBO);
                glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(params), params);
            }
            glDepthFunc(f.WDepth ? (d.DepthLequal ? GL_GEQUAL : GL_GREATER) : (d.DepthLequal ? GL_LEQUAL : GL_LESS));
            glDepthMask(d.DepthWrite ? GL_TRUE : GL_FALSE);
            const GLenum func = d.StencilTest == WideGpuStencil::Always ? GL_ALWAYS :
                d.StencilTest == WideGpuStencil::Equal ? GL_EQUAL : GL_NOTEQUAL;
            glStencilFunc(func, d.StencilRef, d.StencilMask);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            glStencilMask(0xFF);
            if (d.Blend) glEnablei(GL_BLEND, 0);
            else glDisablei(GL_BLEND, 0);
            if (AttrPacked) glColorMaski(1, d.WriteDepthAttr, d.WriteDepthAttr, d.WriteDepthAttr, d.WriteFogAttr);
            else glColorMaski(1, d.WriteDepthAttr, d.WriteFogAttr, GL_FALSE, GL_FALSE);
            glDrawArrays(GL_TRIANGLES, EsPaths ? (GLint)d.First : 0, d.Count);
        }

        // Depth and stencil are not read after this pass: a tiled GPU (Adreno, Mali) then does
        // not write them out from tile memory (W x H x 8 bytes for DEPTH32F_STENCIL8).
        if (Invalidate)
        {
            const GLenum ds = GL_DEPTH_STENCIL_ATTACHMENT;
            Invalidate(GL_FRAMEBUFFER, 1, &ds);
        }

        // The final pass, and the picture read back.
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisablei(GL_BLEND, 0);
        if (ClipControl) ClipControl(LowerLeft, 0x935E); // GL_NEGATIVE_ONE_TO_ONE again
#ifdef __ANDROID__
        // (SSR_WIDE_ZEROCOPY: into the frontend's texture for this slot, when it has one of this
        // size and wants no pixels.)
        ZcTex = ZeroCopy ? ZcBegin(slot) : 0;
        if (ZcTex)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, ZcFBO);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ZcTex, 0);
        }
        else
#endif
        glBindFramebuffer(GL_FRAMEBUFFER, FinalFBO);
        // (Every pixel is drawn again by the triangle below: last frame's need not be loaded
        // into tile memory first.)
        if (Invalidate) Invalidate(GL_FRAMEBUFFER, 1, bufs);
        glDrawBuffers(1, bufs);
        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glUseProgram(FinalProgram);
        glBindVertexArray(EmptyVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ColorTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, AttrTex);
        static const bool noFinal = getenv("SSR_WIDEGPU_NOFINAL") != nullptr, noRead = getenv("SSR_WIDEGPU_NOREAD") != nullptr; // (tests)
        if (!noFinal) glDrawArrays(GL_TRIANGLES, 0, 3);
#ifdef __ANDROID__
        if (ZcTex)
        {
            ZcEnd();
            return true;
        }
#endif
        glBindFramebuffer(GL_READ_FRAMEBUFFER, FinalFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ReadPBO);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        // (OpenGL ES reads back RGBA only: there the final pass has swapped red and blue.)
        glReadPixels(0, 0, noRead ? 1 : W, noRead ? 1 : H, EsPaths ? GL_RGBA : GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glFlush();
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR)
        {
            static int reported = 0;
            if (reported++ < 5) fprintf(stderr, "[widegpu] OpenGL error 0x%x\n", err);
        }
        return true;
    }

    void Fetch(int slot) override
    {
#ifdef __ANDROID__
        // (SSR_WIDE_ZEROCOPY: the picture stays in the texture unless a row has nothing at an end.)
        if (ZcTex && !ZcFetch()) return;
#endif
        u32* out = Slots[slot].data();
        // (SSR_WIDE_PROF=1: how long mapping the picture takes, which waits for the graphics chip
        // to finish it, and how long copying it out takes.)
        static const bool prof = getenv("SSR_WIDE_PROF") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ReadPBO);
        const void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)W * H * 4, GL_MAP_READ_BIT);
        const auto t1 = std::chrono::steady_clock::now();
        if (p) memcpy(out, p, (size_t)W * H * 4);
        else memset(out, 0, (size_t)W * H * 4);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        if (prof)
        {
            const auto t2 = std::chrono::steady_clock::now();
            static double map = 0, copy = 0;
            static u64 n = 0;
            map += std::chrono::duration<double, std::milli>(t1 - t0).count();
            copy += std::chrono::duration<double, std::milli>(t2 - t1).count();
            if (++n % 200 == 0) fprintf(stderr, "[wide] %dx%d picture: %.2f ms to map (waiting for the graphics chip), %.2f ms to copy out\n", W, H, map / n, copy / n);
        }
    }

#ifdef __ANDROID__
    bool ZeroCopy = false; // (set before Init)

    // After Fetch: where the slot's picture is now.
    void ZcPublish(int slot)
    {
        std::lock_guard<std::mutex> lock(ZcMutex);
        ZcSlot& z = Zc[slot];
        z.Drawn = ZcTex;
        z.Drawing = 0;
        z.HasPixels = !ZcTex || ZcPixels;
    }

private:
    // The frontend's texture for the slot (0: none of this size, or pixels wanted: read back as
    // without SSR_WIDE_ZEROCOPY). The graphics chip waits for the frontend's last draws from it
    // (the fence put after the frame that showed it was presented); this thread does not.
    GLuint ZcBegin(int slot)
    {
        GLsync wait = nullptr;
        GLuint tex = 0;
        {
            std::lock_guard<std::mutex> lock(ZcMutex);
            ZcSlot& z = Zc[slot];
            z.Drawn = 0;
            z.HasPixels = false;
            if (!WideZcWantPixels.load() && z.Offered && z.OfferedW == W && z.OfferedH == H)
            {
                tex = z.Drawing = z.Offered;
                wait = z.ReadFence;
                z.ReadFence = nullptr;
            }
        }
        if (wait)
        {
            glWaitSync(wait, 0, GL_TIMEOUT_IGNORED);
            glDeleteSync(wait);
        }
        return tex;
    }

    // After the final pass into the texture: its two outermost columns read back (2 x H
    // pixels), which tell whether MergedFillWideHoles has anything to fill; the fence that
    // Fetch waits for.
    void ZcEnd()
    {
        if (EdgeH != H)
        {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, EdgePBO);
            glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)H * 8, nullptr, GL_STREAM_READ);
            EdgeH = H;
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ZcFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, EdgePBO);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, 1, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glReadPixels(W - 1, 0, 1, H, GL_RGBA, GL_UNSIGNED_BYTE, (void*)((size_t)H * 4));
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        WriteFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR)
        {
            static int reported = 0;
            if (reported++ < 5) fprintf(stderr, "[widegpu] zero copy: OpenGL error 0x%x\n", err);
        }
    }

    // Waits for the picture in the texture. True when the whole picture is wanted in the slot's
    // memory too (a row has nothing drawn at its left or right end): its read-back is then
    // started, for Fetch to copy out as usual.
    bool ZcFetch()
    {
        if (WriteFence)
        {
            GLenum r;
            do r = glClientWaitSync(WriteFence, GL_SYNC_FLUSH_COMMANDS_BIT, 50000000);
            while (r == GL_TIMEOUT_EXPIRED);
            glDeleteSync(WriteFence);
            WriteFence = nullptr;
        }
        bool covered = false;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, EdgePBO);
        if (const u8* e = (const u8*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)H * 8, GL_MAP_READ_BIT))
        {
            covered = true;
            for (int i = 0; i < 2 * H && covered; i++) covered = e[i * 4 + 3] != 0;
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        ZcPixels = !covered;
        // (SSR_WIDE_PROF=1: how many pictures stayed in the texture alone.)
        static const bool prof = getenv("SSR_WIDE_PROF") != nullptr;
        static u64 kept = 0, read = 0;
        if (prof && (covered ? ++kept : ++read) && (kept + read) % 300 == 0)
            fprintf(stderr, "[wide] zero copy: %llu pictures in the texture alone, %llu also read back (a row empty at an end)\n",
                (unsigned long long)kept, (unsigned long long)read);
        if (covered) return false;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ZcFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ReadPBO);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        return true;
    }

    GLuint ZcFBO = 0, EdgePBO = 0, ZcTex = 0;
    int EdgeH = 0;
    GLsync WriteFence = nullptr;
    bool ZcPixels = false;
#endif

private:
    void FreeTargets()
    {
        if (MainFBO) glDeleteFramebuffers(1, &MainFBO);
        if (FinalFBO) glDeleteFramebuffers(1, &FinalFBO);
        const GLuint tex[3] = { ColorTex, AttrTex, FinalTex };
        glDeleteTextures(3, tex);
        if (DepthRB) glDeleteRenderbuffers(1, &DepthRB);
        MainFBO = FinalFBO = ColorTex = AttrTex = FinalTex = DepthRB = 0;
    }

    static GLuint Texture(GLenum format, int w, int h)
    {
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        // (glTexImage2D: glTexStorage2D is OpenGL 4.2, which the loader does not fetch for OpenGL ES.)
        if (format == GL_RG32UI) glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32UI, w, h, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, nullptr);
        else if (format == GL_RGB10_A2UI) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2UI, w, h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV, nullptr);
        else glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        return t;
    }

    void MakeTargets(int w, int h)
    {
        FreeTargets();
        W = w;
        H = h;
        ColorTex = Texture(GL_RGBA8, w, h);
        AttrTex = Texture(AttrPacked ? GL_RGB10_A2UI : GL_RG32UI, w, h);
        FinalTex = Texture(GL_RGBA8, w, h);
        glGenRenderbuffers(1, &DepthRB);
        glBindRenderbuffer(GL_RENDERBUFFER, DepthRB);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, w, h);
        glGenFramebuffers(1, &MainFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, MainFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ColorTex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, AttrTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, DepthRB);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "[widegpu] main framebuffer incomplete\n");
        glGenFramebuffers(1, &FinalFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FinalFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FinalTex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "[widegpu] final framebuffer incomplete\n");
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ReadPBO);
        glBufferData(GL_PIXEL_PACK_BUFFER, (size_t)w * h * 4, nullptr, GL_STREAM_READ);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    GLuint MainProgram = 0, EarlyProgram = 0, FinalProgram = 0;
    GLint DrawLoc[2] { -1, -1 };
    GLuint FrameUBO = 0, DrawUBO = 0, SSBO[4] {}, ReadPBO = 0, EmptyVAO = 0, VertexVAO = 0;
    GLuint PolyTex = 0;    // the polygons as a buffer texture (OpenGL ES: wide_vsh.glsl)
    GLuint BufTex[4] {};   // the texture memory as bytes and 16-bit texels, the palette memory, the atlas (WIDE_TEXBUF)
    GLuint AtlasBuf = 0;
    bool TexBuf = false, AtlasSynced = false;
    bool AttrPacked = false; // SSR_WIDEGPU_ATTR8: the second target RGB10_A2UI (see Init)
    GLuint MainFBO = 0, FinalFBO = 0, ColorTex = 0, AttrTex = 0, FinalTex = 0, DepthRB = 0;
    int W = 0, H = 0;
    ClipControlFn ClipControl = nullptr;
    InvalidateFn Invalidate = nullptr;
    std::vector<u32> Slots[3];
    u64 TexVersion = 0;
};
}

namespace
{
#ifdef __ANDROID__
// A phone's processor has small cores beside its fast ones, and the system likes to put a thread
// that mostly waits on a small one at a low clock. This thread's work is short but the emulation
// waits for it (the picture's draw calls and its copy out of the graphics chip: now and then 6 to
// 11 ms instead of 2, a frame shown late). So it is kept off the smallest cores: those whose top
// clock is the lowest of all, when the cores differ. SSR_WIDEGPU_ANYCORE=1 leaves it alone.
void KeepOffSmallCores()
{
    if (getenv("SSR_WIDEGPU_ANYCORE")) return;
    long top[64] {}, lowest = 0, highest = 0;
    int n = 0;
    for (; n < 64; n++)
    {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", n);
        FILE* f = fopen(path, "r");
        if (!f) break;
        if (fscanf(f, "%ld", &top[n]) != 1) top[n] = 0;
        fclose(f);
        if (!n || top[n] < lowest) lowest = top[n];
        if (top[n] > highest) highest = top[n];
    }
    if (n < 2 || lowest == highest || lowest <= 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    int kept = 0;
    for (int i = 0; i < n; i++)
        if (top[i] > lowest)
        {
            CPU_SET(i, &set);
            kept++;
        }
    if (kept && sched_setaffinity(0, sizeof(set), &set) == 0)
        fprintf(stderr, "[widegpu] its thread keeps to the %d faster cores of %d\n", kept, n);
}
#endif

// WideGpuGL on a thread of its own, its context current there for as long as it runs. Submit
// copies the frame (and the texture memory it is drawn from) and hands it over; the thread draws
// it and copies the picture into Target's memory; Fetch waits for that.
class WideGpuGLWorker : public WideGpuBackend
{
public:
    const char* Name() const override { return "OpenGL"; }
    bool TakesAtlas() const override { return Inner && Inner->TakesAtlas(); }
#ifdef __ANDROID__
    bool ZeroCopy = false; // (set before Start: see WideGpuMakeGL)
#endif

    bool Start(std::string& info)
    {
        Thread = std::thread([this] { Run(); });
        std::unique_lock<std::mutex> lock(Mutex);
        Changed.wait(lock, [this] { return Ready || Failed; });
        info = Info;
        return Ready;
    }

    ~WideGpuGLWorker() override
    {
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Quit = true;
        }
        Changed.notify_all();
        if (Thread.joinable()) Thread.join();
    }

    u32* Target(int slot, int w, int h) override
    {
        // (No OpenGL here: the memory is the processor's. The slot asked for is never the one
        // the thread is writing, which Fetch has waited for before a new one is prepared.)
        return Inner ? Inner->Target(slot, w, h) : nullptr;
    }

    bool Submit(WideGpuFrame& f, int slot) override
    {
        Fetch(PendingSlot); // (one picture at a time)
        Frame.W = f.W;
        Frame.H = f.H;
        Frame.Constants = f.Constants;
        Frame.Vertices = f.Vertices;
        Frame.Polygons = f.Polygons;
        Frame.Draws = f.Draws;
        memcpy(Frame.ClearColor, f.ClearColor, sizeof(Frame.ClearColor));
        Frame.ClearDepth = f.ClearDepth;
        Frame.WDepth = f.WDepth;
        Frame.ClearDepthValue = f.ClearDepthValue;
        Frame.ClearFog = f.ClearFog;
        Frame.ClearStencil = f.ClearStencil;
        Frame.TexVersion = f.TexVersion;
        if (!f.TexVersion || f.TexVersion != CopiedTexVersion)
        {
            memcpy(Tex.data(), f.TexMem, Tex.size());
            memcpy(Pal.data(), f.PalMem, Pal.size());
            CopiedTexVersion = f.TexVersion;
        }
        Frame.TexMem = Tex.data();
        Frame.PalMem = Pal.data();
        // The decoded textures (WideGpuAtlas): this thread decodes into f.Atlas while the worker
        // uploads, so the worker has its own copy, brought up to date with what changed.
        Frame.AtlasOn = f.AtlasOn;
        if (f.AtlasOn)
        {
            const WideGpuAtlas& a = f.Atlas;
            WideGpuAtlas& b = Frame.Atlas;
            if (b.Texels.empty()) b.Texels.resize(WideGpuAtlas::Capacity);
            b.Used = a.Used;
            if (!AtlasCopied)
            {
                memcpy(b.Texels.data(), a.Texels.data(), (size_t)a.Used * 4);
                AtlasCopied = true;
            }
            else
                for (const auto& [first, n] : a.Uploads)
                    memcpy(b.Texels.data() + first, a.Texels.data() + first, (size_t)n * 4);
            b.Uploads = a.Uploads;
        }
        {
            std::lock_guard<std::mutex> lock(Mutex);
            WorkSlot = slot;
            Work = true;
            Finished = false;
        }
        PendingSlot = slot;
        Changed.notify_all();
        // The direction of y the backend draws in, as WideGpuGL's Submit sets it.
        f.Constants.Scale[2] = -1.0f;
        return true;
    }

    void Fetch(int /*slot*/) override
    {
        std::unique_lock<std::mutex> lock(Mutex);
        Changed.wait(lock, [this] { return !Work || Finished || Failed; });
    }

    bool Done(int /*slot*/) override
    {
        std::lock_guard<std::mutex> lock(Mutex);
        return !Work || Finished;
    }

private:
    void Run()
    {
#ifdef __ANDROID__
        KeepOffSmallCores();
#endif
        std::string info;
        bool ok = false;
#ifdef __ANDROID__
        if (ZeroCopy)
        {
            // (A context that shares SDL's textures; without one, the picture is read back.)
            ok = WideGpuGLCreateShared && WideGpuGLCreateShared(info);
            fprintf(stderr, "[widegpu] zero copy (SSR_WIDE_ZEROCOPY): %s%s\n", ok ? "on, " : "off, no context shared with SDL's: ",
                info.c_str());
            if (!ok)
            {
                ZcRelease(false);
                ZeroCopy = false;
            }
        }
        if (!ok)
#endif
        ok = GLThreadRenderer::CreateContext && GLThreadRenderer::CreateContext(info);
        if (ok)
        {
            Inner.reset(new WideGpuGL());
#ifdef __ANDROID__
            Inner->ZeroCopy = ZeroCopy;
#endif
            ok = Inner->Init(info);
            if (!ok) Inner.reset();
        }
        else if (!GLThreadRenderer::CreateContext)
            info = "no OpenGL context in this build";
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Info = info;
            Ready = ok;
            Failed = !ok;
        }
        Changed.notify_all();
        while (ok)
        {
            int slot;
            {
                std::unique_lock<std::mutex> lock(Mutex);
                Changed.wait(lock, [this] { return Quit || (Work && !Finished); });
                if (Quit) break;
                slot = WorkSlot;
            }
            Inner->Submit(Frame, slot);
            Inner->Fetch(slot);
#ifdef __ANDROID__
            if (ZeroCopy) Inner->ZcPublish(slot);
#endif
            {
                std::lock_guard<std::mutex> lock(Mutex);
                Finished = true;
            }
            Changed.notify_all();
        }
        // (Its OpenGL objects belong to this thread's context, which is then let go of, so that
        // another thread can make a new one.)
#ifdef __ANDROID__
        if (ZeroCopy) ZcRelease(Inner != nullptr);
#endif
        Inner.reset();
        if (WideGpuGLRelease) WideGpuGLRelease();
    }

    std::thread Thread;
    std::mutex Mutex;
    std::condition_variable Changed;
    bool Ready = false, Failed = false, Quit = false, Work = false, Finished = false;
    int WorkSlot = 0, PendingSlot = 0;
    std::string Info;
    std::unique_ptr<WideGpuGL> Inner;
    WideGpuFrame Frame;
    std::vector<u8> Tex = std::vector<u8>(512 * 1024), Pal = std::vector<u8>(128 * 1024);
    u64 CopiedTexVersion = 0;
    bool AtlasCopied = false;
};
}

std::unique_ptr<WideGpuBackend> WideGpuMakeGL(std::string& info)
{
    if (!GLThreadRenderer::CreateContext)
    {
        info = "no OpenGL context in this build";
        return nullptr;
    }
    if (WideGpuGLThreaded)
    {
        std::unique_ptr<WideGpuGLWorker> w(new WideGpuGLWorker());
#ifdef __ANDROID__
        // (SSR_WIDE_ZEROCOPY: the first worker made once SDL's context can be shared, the
        // widescreen one: the DS-resolution renderer's worker is made after it.)
        if (WideZcWanted() && WideGpuGLCreateShared)
        {
            std::lock_guard<std::mutex> lock(ZcMutex);
            if (!ZcOn) w->ZeroCopy = ZcOn = true;
        }
#endif
        if (!w->Start(info)) return nullptr;
        return w;
    }
    // One context for every renderer (it stays current on this thread).
    static bool haveContext = false;
    if (!haveContext && !GLThreadRenderer::CreateContext(info)) return nullptr;
    haveContext = true;
    std::unique_ptr<WideGpuGL> r(new WideGpuGL());
    if (!r->Init(info)) return nullptr;
    return r;
}

#ifdef __ANDROID__
// SSR_WIDE_ZEROCOPY (GPU3D_WideGpu.h): what the frontend's thread calls.
bool WideZcWanted()
{
    // (On by default: emulate 10.5 -> 8.7 ms and compose 2.5 -> 0.8 ms a 720-line frame on an
    // Adreno 610. SSR_WIDE_ZEROCOPY=0: the readback.)
    static const bool on = !getenv("SSR_WIDE_ZEROCOPY") || getenv("SSR_WIDE_ZEROCOPY")[0] != '0';
    return on;
}

bool WideZcActive()
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    return ZcOn;
}

void WideZcOffer(int slot, unsigned tex, int w, int h)
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    if (!ZcOn || slot < 0 || slot > 2) return;
    Zc[slot].Offered = tex;
    Zc[slot].OfferedW = w;
    Zc[slot].OfferedH = h;
}

bool WideZcLookup(const u32* pixels, WideZcPicture& out)
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    if (!ZcOn || !pixels) return false;
    for (int s = 0; s < 3; s++)
        if (Zc[s].Pixels == pixels)
        {
            out.Slot = s;
            out.Tex = Zc[s].Drawn;
            out.Pixels = Zc[s].HasPixels;
            return true;
        }
    return false;
}

void WideZcMarkPixels(int slot)
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    if (slot >= 0 && slot < 3) Zc[slot].HasPixels = true;
}

bool WideZcInUse(unsigned tex)
{
    std::lock_guard<std::mutex> lock(ZcMutex);
    for (const ZcSlot& z : Zc)
        if (z.Drawn == tex || z.Drawing == tex) return true;
    return false;
}

unsigned WideZcBoundTexture()
{
    GLint name = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &name);
    return (unsigned)name;
}

void WideZcPresented(int slot)
{
    if (slot < 0 || slot > 2) return;
    // (Flushed at once: the worker's glWaitSync on a fence never sent to the graphics chip could
    // wait for ever. After the present, so nothing of the frame is cut short by it.)
    GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    GLsync old = nullptr;
    {
        std::lock_guard<std::mutex> lock(ZcMutex);
        if (ZcOn)
        {
            old = Zc[slot].ReadFence;
            Zc[slot].ReadFence = f;
            f = nullptr;
        }
    }
    // (A newer fence covers an older one: this context's commands finish in order.)
    if (old) glDeleteSync(old);
    if (f) glDeleteSync(f);
}

// The same for the software renderer's 3D thread (GPU3D_Soft.cpp). While fast forwarding it draws
// one picture in several frames, mostly waits, and was put on a small core: 12 ms a picture
// instead of 4, with the emulation waiting for it.
void SsrKeepThreadOffSmallCores() { KeepOffSmallCores(); }
#endif
}
#endif
