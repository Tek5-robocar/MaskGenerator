using UnityEngine;
using System.IO;
using System;

public class ConfigLoader : MonoBehaviour
{
    public ConfigData Config {get; private set;}
    
    private const string EditorConfigPath = "Assets/generator-config.json";
    private Material[] _materials;

    void Start()
    {

        LoadConfig(EditorConfigPath);
    }

    void LoadConfig(string path)
    {
        try
        {
            string json = File.ReadAllText(path);
            Debug.Log("setting load config");
            Config = JsonUtility.FromJson<ConfigData>(json);
            Debug.Log("loaded config");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load config: {e.Message}");
            Debug.LogError(e.StackTrace);
        }
    }

    [System.Serializable]
    public class Range
    {
        public float min;
        public float max;
    }
    
    [System.Serializable]
    public class ConfigData
    {
        public int nbImage;
        public float rotationRange;
        public float posZoneRadius;
        public float cameraHeight;
        public float cameraAngle;
        public int blurQuantityPercent;
        public int grainQuantityPercent;
        public int colorGradientQuantityPercent;
        public int shapeQuantityPercent;
        public float maxShapeDensity;
        public Range lineWidthMultiplierRange;
        public string versionDirectory;
    }
}