using UnityEngine;
using UnityEngine.UIElements;
using Toggle = UnityEngine.UI.Toggle;

public class RandomPos : MonoBehaviour
{
    public Toggle toggle;
    private float _timer;

    private bool isEnabled;
    public TextField textField;

    private void Start()
    {
        toggle.onValueChanged.AddListener(OnValueChanged);
    }

    private void Update()
    {
        _timer += Time.deltaTime;
    }

    private void OnValueChanged(bool value)
    {
        isEnabled = value;
        textField.SetEnabled(isEnabled);
        if (isEnabled) _timer = 0;
    }
}