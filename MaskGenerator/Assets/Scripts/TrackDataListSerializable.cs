using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

[System.Serializable]
public class TrackData
{
    public string name;
    public float score;

    public TrackData(string name, float score)
    {
        this.name = name;
        this.score = score;
    }
}

[System.Serializable]
public class TrackDataListSerializable
{
    public List<TrackData> tracks = new List<TrackData>();
}

public static class BinarySaveManager
{
    private static readonly string FilePath = Application.persistentDataPath + "/tracks.dat";

    public static void SaveTracks(List<(string, float)> tracks)
    {
        BinaryFormatter formatter = new BinaryFormatter();
        FileStream file = File.Create(FilePath);
        
        TrackDataListSerializable data = new TrackDataListSerializable();
        data.tracks = tracks.ConvertAll(t => new TrackData(t.Item1, t.Item2));
        
        formatter.Serialize(file, data);
        file.Close();
    }

    public static List<(string, float)> LoadTracks()
    {
        if (File.Exists(FilePath))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            FileStream file = File.Open(FilePath, FileMode.Open);
            
            TrackDataListSerializable data = (TrackDataListSerializable)formatter.Deserialize(file);
            file.Close();
            
            return data.tracks.ConvertAll(t => (t.name, t.score));
        }
        return new List<(string, float)>();
    }
}