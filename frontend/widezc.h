// Android, SSR_WIDE_ZEROCOPY=1 (off by default, for comparing the two from one build): the
// widescreen world drawn by the graphics chip (melonDS/src/GPU3D_WideGpu_GL.cpp) goes to the
// screen without being copied. Without it the worker reads each picture back into the
// processor's memory (about 2 ms at 720 lines, plus the graphics chip's time), and gpucompose.h
// uploads it again (about 2.8 ms). With it, the worker's context shares SDL's, and it draws its
// final pass straight into one of three SDL textures, one for each of GPU3D_Wide.cpp's slots.
//
// Who waits for whom:
// - The worker puts a fence after the final pass and waits for it on its own thread (Fetch), so
//   a picture is finished before it is published (WideState::Front). gpucompose.h then binds the
//   texture again in SDL's context before drawing from it, which makes the other context's
//   drawing visible there.
// - After a frame drawn from a slot's texture is presented, a fence goes into SDL's context
//   (WideZcPresented). The worker's graphics chip waits for it (glWaitSync) before it draws into
//   that texture again, three pictures later.
//
// The processor still needs the pixels now and then. The worker reads back both outermost
// columns of every picture (2 x H pixels); when a row has nothing at an end, MergedFillWideHoles
// may fill it, and the whole picture is read back as before. Fast forward, a battle's widened
// screen, frames shown a frame later (`ahead`), screenshots, SSR_CPU_COMPOSE and --log-sections
// ask the worker for pixels (WideZcWantPixels): it then reads the picture back as before. Where
// a picture without pixels is needed after all (the frame after one of those starts), it is read
// back here, on this thread (EnsurePixels), as is the picture under a top screen with backdrop
// in it (gpucompose.h, for MergedWideCovers).
#pragma once
#if defined(__ANDROID__) && defined(SSR_GPU3D)
#include <cstdint>
#include <cstdio>
#include <vector>
#include <SDL.h>
#include "GPU3D_Wide.h"
#include "GPU3D_WideGpu.h"
#include "gpucompose.h"
#include "merged.h"

class WideZeroCopy
{
public:
    // Before a frame runs: textures of the size it is drawn at, offered to the worker, and
    // whether the processor will want the picture's pixels.
    void Prepare(SDL_Renderer* r, int w, int h, bool wantPixels, uint32_t frame)
    {
        melonDS::WideZcWantPixels = wantPixels;
        if (!r || w <= 0 || h <= 0 || !melonDS::WideZcActive()) return;
        for (int s = 0; s < 3; s++)
        {
            Tex& t = Cur[s];
            if (!t.t || t.w != w || t.h != h)
            {
                if (t.t)
                {
                    t.retired = frame;
                    Old.push_back(t);
                }
                t = Make(r, w, h);
            }
            if (t.t) melonDS::WideZcOffer(s, t.name, w, h);
        }
        // Textures of an earlier size, once no slot's picture is in them and the worker is not
        // drawing into them, and some frames on (a frame waiting to be shown may use one).
        for (size_t i = 0; i < Old.size();)
            if (frame - Old[i].retired > 8 && !melonDS::WideZcInUse(Old[i].name))
            {
                SDL_DestroyTexture(Old[i].t);
                Old.erase(Old.begin() + i);
            }
            else i++;
    }

    // The picture at `pixels` (WideState::Front), for readers on the processor: read back from
    // its texture into the slot's memory if it is there only. (Nothing else writes that memory
    // until the worker draws into the slot again, two pictures on.) False if it could not be.
    bool EnsurePixels(SDL_Renderer* r, const uint32_t* pixels, int w, int h)
    {
        melonDS::WideZcPicture p;
        if (!pixels || !melonDS::WideZcLookup(pixels, p) || p.Pixels || !p.Tex) return true;
        SDL_Texture* t = Find(p.Tex, w, h);
        if (t && GpuReadTexture(r, t, const_cast<uint32_t*>(pixels), w, h))
        {
            melonDS::WideZcMarkPixels(p.Slot);
            ReadBacks++;
            return true;
        }
        static int reported = 0;
        if (reported++ < 5) fprintf(stderr, "[wide] zero copy: the picture's texture is gone or unreadable\n");
        return false;
    }

    // A picture about to be drawn by gpucompose.h, when it is in one of the textures.
    void Attach(WideImage& wide)
    {
        melonDS::WideZcPicture p;
        if (!wide.pixels || wide.placed || !melonDS::WideZcLookup(wide.pixels, p) || !p.Tex) return;
        SDL_Texture* t = Find(p.Tex, wide.w, wide.h);
        if (!t) return;
        wide.zcTex = t;
        wide.zcSlot = p.Slot;
        wide.zcStale = !p.Pixels;
    }

    // Right after SDL_RenderPresent (SDL's context current): the fence after the frame's draws
    // from the picture's texture.
    static void Presented(const WideImage& wide)
    {
        if (wide.zcTex) melonDS::WideZcPresented(wide.zcSlot);
    }

    // Before the worker is stopped (WORLD ON GPU turned off): the picture on show read back into
    // its slot's memory if it is only in a texture, for GPU3D_Wide.cpp to keep (WideReleaseGpu).
    void BeforeStop(SDL_Renderer* r) { EnsurePixels(r, melonDS::Wide.Front, melonDS::Wide.FrontW, melonDS::Wide.FrontH); }

    unsigned ReadBacks = 0;

private:
    struct Tex
    {
        SDL_Texture* t = nullptr;
        unsigned name = 0;
        int w = 0, h = 0;
        uint32_t retired = 0;
    };

    // A texture the renderer can draw into (for GpuReadTexture), ARGB8888 like the picture: the
    // final pass of OpenGL ES writes red and blue swapped, which is how SDL keeps ARGB8888 in an
    // OpenGL RGBA texture. Its OpenGL name, for the worker, from binding it.
    static Tex Make(SDL_Renderer* r, int w, int h)
    {
        Tex t;
        t.t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, w, h);
        if (!t.t) return t;
        SDL_GL_BindTexture(t.t, nullptr, nullptr);
        t.name = melonDS::WideZcBoundTexture();
        SDL_GL_UnbindTexture(t.t);
        if (!t.name)
        {
            fprintf(stderr, "[wide] zero copy: no OpenGL name for SDL's texture (%s)\n", SDL_GetError());
            SDL_DestroyTexture(t.t);
            t.t = nullptr;
            return t;
        }
        t.w = w;
        t.h = h;
        return t;
    }

    SDL_Texture* Find(unsigned name, int w, int h) const
    {
        for (const Tex& t : Cur)
            if (t.t && t.name == name && t.w == w && t.h == h) return t.t;
        for (const Tex& t : Old)
            if (t.t && t.name == name && t.w == w && t.h == h) return t.t;
        return nullptr;
    }

    Tex Cur[3];
    std::vector<Tex> Old;
};
#endif
