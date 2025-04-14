using UnityEngine;

public class TextureBlurrer
{
    public static Texture2D BlurTexture(Texture2D original, float blurPercent)
    {
        blurPercent = Mathf.Clamp(blurPercent, 0, 100);
        
        int radius = Mathf.RoundToInt(blurPercent / 10f);
        if (radius == 0) return original;

        Texture2D blurred = new Texture2D(original.width, original.height);
        float[,] kernel = GenerateGaussianKernel(radius);
        int kernelSize = kernel.GetLength(0);

        for (int x = 0; x < original.width; x++)
        {
            for (int y = 0; y < original.height; y++)
            {
                Color blurredColor = ApplyKernel(original, x, y, kernel, kernelSize);
                blurred.SetPixel(x, y, blurredColor);
            }
        }

        blurred.Apply();
        return blurred;
    }

    private static float[,] GenerateGaussianKernel(int radius)
    {
        int size = radius * 2 + 1;
        float[,] kernel = new float[size, size];
        float sigma = radius / 2.0f;
        float twoSigmaSquare = 2.0f * sigma * sigma;
        float piSigma = 1.0f / (Mathf.PI * twoSigmaSquare);
        float sum = 0;

        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                float distance = i * i + j * j;
                kernel[i + radius, j + radius] = piSigma * Mathf.Exp(-distance / twoSigmaSquare);
                sum += kernel[i + radius, j + radius];
            }
        }

        // Normalize kernel values
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                kernel[i, j] /= sum;
            }
        }

        return kernel;
    }

    private static Color ApplyKernel(Texture2D texture, int x, int y, float[,] kernel, int kernelSize)
    {
        Color sumColor = Color.black;
        int radius = kernelSize / 2;

        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                int nx = Mathf.Clamp(x + i, 0, texture.width - 1);
                int ny = Mathf.Clamp(y + j, 0, texture.height - 1);

                Color color = texture.GetPixel(nx, ny);
                sumColor += color * kernel[i + radius, j + radius];
            }
        }

        return sumColor;
    }
}
