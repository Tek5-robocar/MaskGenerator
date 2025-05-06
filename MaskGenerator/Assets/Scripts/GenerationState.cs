using TMPro;
using UnityEngine;

public class GenerationState : MonoBehaviour
{
    public TextMeshProUGUI text;
    public ConfigLoader config;
    
    private float _startTime;

    public int NbGeneration { get; set; } = 0;

    void Start()
    {
        _startTime = Time.time;
    }
    
    void Update()
    {
        float now = Time.time;
        text.text = $"{NbGeneration}/{config.Config.nbImage} in {now - _startTime}ms | approximately {(now - _startTime) / NbGeneration * (config.Config.nbImage - NbGeneration)}ms left";            
    }
}
