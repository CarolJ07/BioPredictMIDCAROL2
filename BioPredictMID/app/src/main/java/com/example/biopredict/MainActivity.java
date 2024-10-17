package com.example.biopredict;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    EditText inputFieldRBC;
    EditText inputFieldHCT;
    EditText inputFieldMCV;

    Button predictBtn;

    TextView resultTV;

    Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        inputFieldRBC = findViewById(R.id.editTextRBC);
        inputFieldHCT = findViewById(R.id.editTextHCT);
        inputFieldMCV = findViewById(R.id.editTextMCV);

        predictBtn = findViewById(R.id.button);
        resultTV = findViewById(R.id.textView);

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String rbcInput = inputFieldRBC.getText().toString();
                String hctInput = inputFieldHCT.getText().toString();
                String mcvInput = inputFieldMCV.getText().toString();

                // Valores mínimos y máximos usados durante el entrenamiento
                float minRBC = 2.77f;
                float maxRBC = 5.95f;

                float minHCT = 24.2f;
                float maxHCT = 316.0f;

                float minMCV = 62.1f;
                float maxMCV = 106.2f;

                // Normalización de los valores de entrada
                float rbcValue = (Float.parseFloat(rbcInput) - minRBC) / (maxRBC - minRBC);
                float hctValue = (Float.parseFloat(hctInput) - minHCT) / (maxHCT - minHCT);
                float mcvValue = (Float.parseFloat(mcvInput) - minMCV) / (maxMCV - minMCV);

                float[][] inputs = new float[1][3];
                inputs[0][0] = rbcValue;
                inputs[0][1] = hctValue;
                inputs[0][2] = mcvValue;

                // Realiza la inferencia
                float result = doInference(inputs);  // Este es el valor normalizado de HGB

                // Desnormalización del resultado
                float minHGB = 0.4f;
                float maxHGB = 16.7f;

                float hgbResult = result * (maxHGB - minHGB) + minHGB;

                // Mostrar el resultado desnormalizado
                resultTV.setText("Result: " + hgbResult);
            }
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    public float doInference(float[][] input) {
        float[][] output = new float[1][1];
        interpreter.run(input, output);
        return output[0][0];
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("MIDTERM_linear.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }
}
