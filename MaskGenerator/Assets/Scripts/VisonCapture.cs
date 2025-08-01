using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using Debug = UnityEngine.Debug;
using File = UnityEngine.Windows.File;
using Random = UnityEngine.Random;
using MotionBlur = UnityEngine.Rendering.PostProcessing.MotionBlur;

public class VisonCapture : MonoBehaviour
{
    public GameObject tracks;
    public ConfigLoader configLoader;
    public CentralLine centralLine;
    public Camera vision;
    public GameObject blackScreen;
    public Light gameLight;
    public Shader standardShader;
    public Shader outlineFillShader;
    public PostProcessVolume postProcessVolume;
    public Material roadMaterial;
    public GenerationState generationState;
    private readonly string _datasetDirectory = "Dataset";
    private readonly string _imageDirectory = "Images";
    private readonly string _maskDirectory = "Masks";
    private readonly float redLaneWidth = 10f;

    private readonly float whiteThreshold = 0.5f;
    private Texture2D _camTexture;

    private ColorGrading _colorGrading;

    private Color _colorGradingFilterColor;

    private int _densityPropertyID;

    private int _existingFiles;
    private Grain _grain;

    private float _lenghtOffset;
    private LineRenderer[] _lineList;
    private MotionBlur _motionBlur;
    private Outline[] _outlineList;
    private int _posIndex;
    private DateTime _startTime;
    private Rect _textureRect;
    private float _totalLength;


    private void Start()
    {
        _startTime = DateTime.Now;
        for (var j = 0; j < tracks.transform.childCount; j++)
            centralLine.SetTrack(tracks.transform.GetChild(j).gameObject);

        Init();
        var cameraRenderTexture =
            new RenderTexture(configLoader.Config.imageWidth, configLoader.Config.imageHeight, 24);
        vision.targetTexture = cameraRenderTexture;
        _camTexture = new Texture2D(vision.pixelWidth, vision.pixelHeight, TextureFormat.RGB24, false);
        _textureRect = new Rect(0, 0, vision.pixelWidth, vision.pixelHeight);

        _outlineList = tracks.GetComponentsInChildren<Outline>();
        _lineList = tracks.GetComponentsInChildren<LineRenderer>();

        _densityPropertyID = Shader.PropertyToID("_MaxDensity");

        if (Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _imageDirectory)) &&
            Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _maskDirectory)))
        {
            var imageDirInfo = new DirectoryInfo(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory,
                _imageDirectory));
            var maskDirInfo = new DirectoryInfo(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory,
                _maskDirectory));
            var images = imageDirInfo.GetFiles();
            var masks = maskDirInfo.GetFiles();
            _existingFiles = Math.Min(images.Length, masks.Length);
            Debug.Log($"Will expand after the {_existingFiles} already existing pairs");
        }
    }

    private void Update()
    {
        if (_posIndex >= configLoader.Config.nbImage)
        {
#if UNITY_EDITOR
            Debug.Log($"Finished in {DateTime.Now - _startTime}");
            Process.Start(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory));
            EditorApplication.isPlaying = false;
#endif
            return;
        }

        var lenght = _lenghtOffset * _posIndex++;
        var localLenght = 0f;

        for (var j = 0; j < tracks.transform.childCount; j++)
        {
            var track = tracks.transform.GetChild(j).gameObject;
            for (var k = 0; k < track.transform.childCount; k++) track.SetActive(false);
        }

        for (var j = 0; j < tracks.transform.childCount; j++)
        {
            var track = tracks.transform.GetChild(j).gameObject;
            for (var k = 0; k < track.transform.childCount; k++)
            {
                track.SetActive(true);
                var trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (var i = 0; i < trackLineRenderer.positionCount; i++)
                {
                    if (lenght >= localLenght && lenght <= localLenght + trackLineRenderer.GetPosition(i).magnitude)
                    {
                        SetPosition(trackLineRenderer, i, localLenght, lenght);
                        Capture();
                        return;
                    }

                    localLenght += trackLineRenderer.GetPosition(i).magnitude;
                }

                track.SetActive(false);
            }
        }
    }

    private void Init()
    {
        for (var j = 0; j < tracks.transform.childCount; j++)
        {
            var track = tracks.transform.GetChild(j).gameObject;
            for (var k = 0; k < track.transform.childCount; k++)
            {
                var trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (var i = 0; i < trackLineRenderer.positionCount; i++)
                    _totalLength += trackLineRenderer.GetPosition(i).magnitude;
            }
        }

        _lenghtOffset = _totalLength / configLoader.Config.nbImage;
        Debug.Log($"Total lenght: {_totalLength}, offset: {_lenghtOffset}");

        if (_colorGrading == null) _colorGrading = postProcessVolume.profile.AddSettings<ColorGrading>();
        postProcessVolume.profile.TryGetSettings(out _colorGrading);

        if (_grain == null) _grain = postProcessVolume.profile.AddSettings<Grain>();
        postProcessVolume.profile.TryGetSettings(out _grain);

        if (_motionBlur == null) _motionBlur = postProcessVolume.profile.AddSettings<MotionBlur>();
        postProcessVolume.profile.TryGetSettings(out _motionBlur);
    }

    private void SetPosition(LineRenderer trackLineRenderer, int i, float localLength, float totalLength)
    {
        var A = trackLineRenderer.GetPosition(i > 0 ? i - 1 : 0);
        var B = trackLineRenderer.GetPosition(i);

        var segmentDistance = Vector3.Distance(A, B);
        var t = (localLength + trackLineRenderer.GetPosition(i).magnitude - totalLength) / segmentDistance;
        t = Mathf.Clamp01(t);

        var C = Vector3.Lerp(A, B, t);
        C.y += Random.Range(configLoader.Config.cameraHeightRange.min, configLoader.Config.cameraHeightRange.max);
        C.x += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        C.z += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        transform.position = C;

        Vector3 forwardDirection;

        if (i < trackLineRenderer.positionCount - 1)
        {
            var nextPoint = trackLineRenderer.GetPosition(i + 1);
            forwardDirection = (nextPoint - B).normalized;
        }
        else
        {
            forwardDirection = (B - A).normalized;
        }

        if (forwardDirection != Vector3.zero)
        {
            var lookRotation = Quaternion.LookRotation(forwardDirection);
            var xRotation =
                Quaternion.Euler(
                    Random.Range(configLoader.Config.cameraAngleRange.min, configLoader.Config.cameraAngleRange.max),
                    Random.Range(-configLoader.Config.rotationRange, configLoader.Config.rotationRange), 0f);
            transform.rotation = lookRotation * xRotation;
        }
    }

    private void SetPostProcessing()
    {
        if (Random.Range(0, 100) <= configLoader.Config.colorGradientQuantityPercent)
        {
            _colorGrading.enabled.Override(true);
            _colorGrading.postExposure.Override(Random.Range(-5f, 5f));
            _colorGradingFilterColor.r = Random.Range(0.8f, 1.0f);
            _colorGradingFilterColor.g = Random.Range(0.8f, 1.0f);
            _colorGradingFilterColor.b = Random.Range(0.8f, 1.0f);
            _colorGrading.colorFilter.Override(_colorGradingFilterColor);
        }

        if (Random.Range(0, 100) <= configLoader.Config.grainQuantityPercent)
        {
            _grain.enabled.Override(true);
            if (Random.Range(0, 100) <= configLoader.Config.coloredGrainQuantityPercent)
                _grain.colored.Override(true);
            else
                _grain.colored.Override(false);
            _grain.intensity.Override(Random.Range(0f, 1f));
            _grain.size.Override(Random.Range(0.3f, 3f));
            _grain.lumContrib.Override(Random.Range(0f, 1f));
        }

        if (Random.Range(0, 100) <= configLoader.Config.blurQuantityPercent)
        {
            _motionBlur.enabled.Override(true);
            _motionBlur.shutterAngle.Override(Random.Range(0f, 360f));
            _motionBlur.sampleCount.Override(Random.Range(4, 32));
        }

        SetRandomMaxDynamicShapesDensity();
        SetLineRendererWidthMultiplier();
    }

    private void SetRandomMaxDynamicShapesDensity()
    {
        if (Random.Range(0, 100) <= configLoader.Config.shapeQuantityPercent)
            roadMaterial.SetFloat(_densityPropertyID, Random.Range(0f, configLoader.Config.maxShapeDensity));
        else
            roadMaterial.SetFloat(_densityPropertyID, 0f);
    }

    private void SetLineRendererWidthMultiplier()
    {
        foreach (var lineRenderer in _lineList)
            lineRenderer.widthMultiplier = Random.Range(configLoader.Config.lineWidthMultiplierRange.min,
                configLoader.Config.lineWidthMultiplierRange.max);
    }

    private void UnsetLineRendererWidthMultiplier()
    {
        foreach (var lineRenderer in _lineList) lineRenderer.widthMultiplier = 0.5f;
    }

    private void UnsetPostProcessing()
    {
        _colorGrading.enabled.Override(false);
        _grain.enabled.Override(false);
        _motionBlur.enabled.Override(false);
        if (configLoader.Config.fixMaskLineWidth) UnsetLineRendererWidthMultiplier();
    }

    private void CaptureCamera(bool blackScreenState, string subPath)
    {
        if (!Directory.Exists(_datasetDirectory)) Directory.CreateDirectory(_datasetDirectory);

        if (!Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory)))
            Directory.CreateDirectory(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory));

        if (!Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, subPath)))
            Directory.CreateDirectory(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, subPath));

        blackScreen.SetActive(blackScreenState);

        if (!blackScreenState)
            SetPostProcessing();
        else
            UnsetPostProcessing();

        // if (!blackScreenState)
        // {
        //     foreach (Outline outline in _outlineList)
        //     {
        //         outline.outlineFillMaterial.shader = standardShader;
        //     }
        // }
        // else
        // {
        //     foreach (Outline outline in _outlineList)
        //     {
        //         outline.outlineFillMaterial.shader = outlineFillShader;
        //     }
        // }

        RenderTexture.active = vision.targetTexture;
        vision.Render();
        _camTexture.ReadPixels(_textureRect, 0, 0);
        _camTexture.Apply();
        byte[] imageBytes;
        if (blackScreenState)
        {
            var cleanedImage = GetMainLane(_camTexture);
            imageBytes = cleanedImage.EncodeToPNG();
        }
        else
        {
            imageBytes = _camTexture.EncodeToPNG();
        }

        File.WriteAllBytes(
            Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, subPath,
                $"{_posIndex + _existingFiles}.png"), imageBytes);
    }

    private void Capture()
    {
        CaptureCamera(false, _imageDirectory);
        CaptureCamera(true, _maskDirectory);
        generationState.NbGeneration++;
    }

    private Texture2D GetMainLane(Texture2D inputTexture)
    {
        var width = inputTexture.width;
        var height = inputTexture.height;

        var outputTexture = new Texture2D(width, height);
        outputTexture.filterMode = FilterMode.Point;

        var pixels = inputTexture.GetPixels();
        var newPixels = Enumerable.Repeat(Color.black, pixels.Length).ToArray();

        var numberRay = 20;
        var fieldView = 180;
        var angleOffset = fieldView / (numberRay - 1);
        var stepSize = 1;

        for (var k = 0; k < numberRay; k++)
        {
            var x = width / 2f;
            var y = height - 1f;

            while (x >= 0 && x < inputTexture.width && y >= 0 && y < inputTexture.height)
            {
                var angle = k * angleOffset * Mathf.PI / 180 +
                            angleOffset * Mathf.PI / 180 * ((180 - fieldView) / angleOffset / 2);
                var roundedX = Mathf.FloorToInt(x);
                var roundedY = Mathf.FloorToInt(y);
                var pixelColor = inputTexture.GetPixel(roundedX, inputTexture.height - 1 - roundedY);
                // newPixels[roundedX + (inputTexture.height - 1 - roundedY) * width] = Color.blue;
                if (pixelColor.r > 0.9f && pixelColor.g > 0.9f && pixelColor.b > 0.9f)
                {
                    var lanePixels = FloodFill(inputTexture, roundedX, inputTexture.height - 1 - roundedY, 3);
                    
                    foreach (var pixel in lanePixels)
                    {
                        var index = pixel.y * width + pixel.x;
                        newPixels[index] = Color.white;
                    }

                    break;
                }

                x += stepSize * Mathf.Cos(angle);
                y -= stepSize * Mathf.Sin(angle);
            }
        }
        outputTexture.SetPixels(newPixels);
        outputTexture.Apply();
        FillLaneUnderLines(outputTexture);
        return outputTexture;
    }

    private void FillLaneUnderLines(Texture2D texture)
    {
        int maxHeight = texture.height - 1;
        bool maxHeightReached = false;
        for (;maxHeight > 0; maxHeight--)
        {
            for (int x = 0; x < texture.width; x++)
            {
                if (texture.GetPixel(x, maxHeight).b > 0.1)
                {
                    maxHeightReached = true;
                    break;
                }
            }

            if (maxHeightReached) break;
        }        
        int middle = texture.width / 2;
        for (int y = 0; y < maxHeight && y < texture.height; y++)
        {
            int middleOffset = 0;
            for (int x = middle; x < texture.width; x++, middleOffset++)
            {
                if (texture.GetPixel(x, y).b > 0.1) break;
                texture.SetPixel(x, y, Color.red);
            }
            for (int x = middle; x > 0; x--,  middleOffset--)
            {
                if (texture.GetPixel(x, y).b > 0.1) break;
                texture.SetPixel(x, y, Color.red);
            }
            middle += middleOffset;
        }

        for (int y = 0; y < texture.height; y++)
        {
            for (int x = 0; x < texture.width; x++)
            {
                if (texture.GetPixel(x, y).b > 0.1) texture.SetPixel(x, y, Color.black);
                else if (texture.GetPixel(x, y).r > 0.1) texture.SetPixel(x, y, Color.white);
            }
        }
        texture.Apply();
    }

    private HashSet<Vector2Int> FloodFill(Texture2D image, int startX, int startY, int z)
    {
        var visited = new HashSet<Vector2Int>();
        var queue = new Queue<Vector2Int>();
        var width = image.width;
        var height = image.height;

        var start = new Vector2Int(startX, startY);
        if (!IsValidPixel(start, width, height) || image.GetPixel(startX, startY) != Color.white) return visited;

        queue.Enqueue(start);
        visited.Add(start);

        // Directions for 4-connectivity (up, down, left, right)
        Vector2Int[] directions =
        {
            new(0, 1),
            new(0, -1),
            new(-1, 0),
            new(1, 0)
        };

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();

            foreach (var dir in directions)
            {
                var next = current;
                var steps = 0;
                var foundWhite = false;
                var target = current;

                // Look ahead up to z pixels in the current direction
                for (var i = 1; i <= z; i++)
                {
                    next = current + dir * i;
                    if (!IsValidPixel(next, width, height)) break; // Stop if we hit the image boundary

                    if (image.GetPixel(next.x, next.y) == Color.white)
                    {
                        foundWhite = true;
                        target = next;
                        break; // Found a white pixel within z steps
                    }
                }

                if (foundWhite && !visited.Contains(target))
                {
                    // Add all pixels (including black ones) from current to target to visited
                    for (var i = 1; i <= (target - current).magnitude; i++)
                    {
                        var intermediate = current + dir * i;
                        if (IsValidPixel(intermediate, width, height) && !visited.Contains(intermediate))
                            visited.Add(intermediate);
                    }

                    queue.Enqueue(target);
                    visited.Add(target);
                }
                else if (IsValidPixel(next, width, height) && !visited.Contains(next) &&
                         image.GetPixel(next.x, next.y) == Color.white)
                {
                    // No gap, directly adjacent white pixel
                    queue.Enqueue(next);
                    visited.Add(next);
                }
            }
        }

        return visited;
    }

    private bool IsValidPixel(Vector2Int pixel, int width, int height)
    {
        return pixel.x >= 0 && pixel.x < width && pixel.y >= 0 && pixel.y < height;
    }

}