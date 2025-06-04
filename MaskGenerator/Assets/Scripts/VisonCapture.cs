using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using File = UnityEngine.Windows.File;
using Random = UnityEngine.Random;
using UnityEngine.Rendering.PostProcessing;
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

    private float _lenghtOffset;
    private float _totalLength = 0;
    private int _posIndex = 0;
    private readonly string _imageDirectory = "Images";
    private readonly string _maskDirectory = "Masks";
    private readonly string _datasetDirectory = "Dataset";
    private DateTime _startTime;
    private Texture2D _camTexture;
    private Rect _textureRect;
    private Outline[] _outlineList;
    private LineRenderer[] _lineList;
    
    private ColorGrading _colorGrading;
    private MotionBlur _motionBlur;
    private Grain _grain;
    
    private int _densityPropertyID;

    private int _existingFiles = 0;
    
    private Color _colorGradingFilterColor = new Color(); 

    
    private void Start()
    {
        _startTime = DateTime.Now;
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            centralLine.SetTrack(tracks.transform.GetChild(j).gameObject);
        }
        
        Init();

        _camTexture = new Texture2D(vision.pixelWidth, vision.pixelHeight, TextureFormat.RGB24, false);
        _textureRect = new Rect(0, 0, vision.pixelWidth, vision.pixelHeight);

        _outlineList = tracks.GetComponentsInChildren<Outline>();
        _lineList = tracks.GetComponentsInChildren<LineRenderer>();

        _densityPropertyID = Shader.PropertyToID("_MaxDensity");

        if (Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _imageDirectory)) && Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _maskDirectory)))
        {
            var imageDirInfo = new DirectoryInfo(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _imageDirectory));
            var maskDirInfo = new DirectoryInfo(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory, _maskDirectory));
            var images = imageDirInfo.GetFiles();
            var masks = maskDirInfo.GetFiles();
            _existingFiles = Math.Min(images.Length, masks.Length);
            Debug.Log($"Will expand after the {_existingFiles} already existing pairs");
        }
    }
    
    private void Init() 
    {
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            GameObject track = tracks.transform.GetChild(j).gameObject;
            for (int k = 0; k < track.transform.childCount; k++)
            {
                LineRenderer trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (int i = 0; i < trackLineRenderer.positionCount; i++)
                {
                    _totalLength += trackLineRenderer.GetPosition(i).magnitude;
                }
            }
        }
        _lenghtOffset = _totalLength / configLoader.Config.nbImage;
        Debug.Log($"Total lenght: {_totalLength}, offset: {_lenghtOffset}");
        
        if(_colorGrading == null)
        {
            _colorGrading = postProcessVolume.profile.AddSettings<ColorGrading>();
        }
        postProcessVolume.profile.TryGetSettings(out _colorGrading);
        
        if(_grain == null)
        {
            _grain = postProcessVolume.profile.AddSettings<Grain>();
        }
        postProcessVolume.profile.TryGetSettings(out _grain);
        
        if(_motionBlur == null)
        {
            _motionBlur = postProcessVolume.profile.AddSettings<MotionBlur>();
        }
        postProcessVolume.profile.TryGetSettings(out _motionBlur);
    }

    private void SetPosition(LineRenderer trackLineRenderer, int i, float localLength, float totalLength)
    {
        Vector3 A = trackLineRenderer.GetPosition(i > 0 ? i - 1 : 0);
        Vector3 B = trackLineRenderer.GetPosition(i);

        float segmentDistance = Vector3.Distance(A, B);
        float t = (localLength + trackLineRenderer.GetPosition(i).magnitude - totalLength) / segmentDistance;
        t = Mathf.Clamp01(t);

        Vector3 C = Vector3.Lerp(A, B, t);
        C.y += Random.Range(configLoader.Config.cameraHeightRange.min, configLoader.Config.cameraHeightRange.max);
        C.x += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        C.z += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        transform.position = C;

        Vector3 forwardDirection;
    
        if (i < trackLineRenderer.positionCount - 1)
        {
            Vector3 nextPoint = trackLineRenderer.GetPosition(i + 1);
            forwardDirection = (nextPoint - B).normalized;
        }
        else
        {
            forwardDirection = (B - A).normalized;
        }

        if (forwardDirection != Vector3.zero)
        {
            Quaternion lookRotation = Quaternion.LookRotation(forwardDirection);
            Quaternion xRotation = Quaternion.Euler(Random.Range(configLoader.Config.cameraAngleRange.min, configLoader.Config.cameraAngleRange.max), Random.Range(-configLoader.Config.rotationRange, configLoader.Config.rotationRange), 0f);
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
        {
            roadMaterial.SetFloat(_densityPropertyID, Random.Range(0f, configLoader.Config.maxShapeDensity));
        }
        else
        {
            roadMaterial.SetFloat(_densityPropertyID, 0f);
        }
    }

    private void SetLineRendererWidthMultiplier()
    {
        foreach (LineRenderer lineRenderer in _lineList)
        {
            lineRenderer.widthMultiplier = Random.Range(configLoader.Config.lineWidthMultiplierRange.min, configLoader.Config.lineWidthMultiplierRange.max);
        }
    }
    
    private void UnsetLineRendererWidthMultiplier()
    {
        foreach (LineRenderer lineRenderer in _lineList)
        {
            lineRenderer.widthMultiplier = 0.5f;
        }
    }
    
    private void UnsetPostProcessing()
    {
        _colorGrading.enabled.Override(false);
        _grain.enabled.Override(false);
        _motionBlur.enabled.Override(false);
        if (configLoader.Config.fixMaskLineWidth)
        {
            UnsetLineRendererWidthMultiplier();
        }
    }

    private void CaptureCamera(bool blackScreenState, string subPath)
    {
        if (!Directory.Exists(_datasetDirectory))
        {
            Directory.CreateDirectory(_datasetDirectory);
        }
        
        if (!Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory)))
        {
            Directory.CreateDirectory(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory));
        }

        if (!Directory.Exists(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath)))
        {
            Directory.CreateDirectory(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath));
        }
        
        blackScreen.SetActive(blackScreenState);

        if (!blackScreenState)
        {
            SetPostProcessing();
        }
        else
        {
            UnsetPostProcessing();
        }
        
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
        
        byte[] imageBytes = _camTexture.EncodeToPNG();
        File.WriteAllBytes(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath, $"{_posIndex + _existingFiles}.png"), imageBytes);
    }

    private void Capture()
    {
        CaptureCamera(false, _imageDirectory);
        CaptureCamera(true, _maskDirectory);
        generationState.NbGeneration++;
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
        
        float lenght = _lenghtOffset * _posIndex++;
        float localLenght = 0f;
        
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            GameObject track = tracks.transform.GetChild(j).gameObject;
            for (int k = 0; k < track.transform.childCount; k++)
            {
                track.SetActive(false);
            }
        }
        
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            GameObject track = tracks.transform.GetChild(j).gameObject;
            for (int k = 0; k < track.transform.childCount; k++)
            {
                track.SetActive(true);
                LineRenderer trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (int i = 0; i < trackLineRenderer.positionCount; i++)
                {
                    if (lenght >= localLenght && lenght <= localLenght + trackLineRenderer.GetPosition(i).magnitude)
                    {
                        SetPosition(trackLineRenderer,  i, localLenght, lenght);
                        Capture();
                        return;
                    }
                    localLenght += trackLineRenderer.GetPosition(i).magnitude;
                }
                track.SetActive(false);
            }
        }
    }
}