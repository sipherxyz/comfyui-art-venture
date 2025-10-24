# ArtVenture Custom Nodes

A comprehensive set of custom nodes for ComfyUI, focusing on utilities for image processing, JSON manipulation, model operations and working with object via URLs

### Image Nodes

#### LoadImageFromUrl

Loads images from URLs.

**Inputs:**

- `image`: List of URLs or base64 image data, separated by new lines
- `keep_alpha_channel`: Preserve alpha channel
- `output_mode`: List or batch output. Use `List` if you have different resolutions.

![load image from url](https://github.com/user-attachments/assets/9da4840c-925e-4e0c-984a-5412282aee79)

### JSON Nodes

#### LoadJsonFromUrl

Loads JSON data from URLs.

**Inputs:**

- `url`: JSON URL
- `print_to_console`: Print JSON to console

#### LoadJsonFromText

Loads JSON data from text.

**Inputs:**

- `data`: JSON text
- `print_to_console`: Print JSON to console

#### Get<\*>FromJson

Includes `GetObjectFromJson`, `GetTextFromJson`, `GetFloatFromJson`, `GetIntFromJson`, `GetBoolFromJson`.

Use key format `key.[index].subkey.[sub_index]` to access nested objects.

![get data from json](https://github.com/user-attachments/assets/a71793d6-9661-441c-a15c-66b2dcaa7972)

### Utility Nodes

#### StringToNumber

Converts strings to numbers.

**Inputs:**

- `string`: Input string
- `rounding`: Rounding method

#### TextRandomMultiline

Randomizes the order of lines in a multiline string.

**Inputs:**

- `text`: Input text
- `amount`: Number of lines to randomize
- `seed`: Random seed

![text random multiline](https://github.com/user-attachments/assets/86f811e3-579e-4ccc-81a3-e216cd851d3c)

#### TextSwitchCase

Switch between multiple cases based on a condition.

**Inputs:**

- `switch_cases`: Switch cases, separated by new lines
- `condition`: Condition to switch on
- `default_value`: Default value when no condition matches
- `delimiter`: Delimiter between case and value, default is `:`

The `switch_cases` format is `case<delimiter>value`, where `case` is the condition to match and `value` is the value to return when the condition matches. You can have new lines in the value to return multiple lines.

![text switch case](https://github.com/user-attachments/assets/4c5450a8-6a3a-4d3c-8c2a-c6e3a33cb95f)

### Inpainting Nodes

#### PrepareImageAndMaskForInpaint

Prepares images and masks for inpainting operations. It's to mimic the behavior of the inpainting in A1111.

**Inputs:**

- `image`: Input image tensor
- `mask`: Input mask tensor
- `mask_blur`: Blur amount for mask (0-64)
- `inpaint_masked`: Whether to inpaint only the masked regions, otherwise it will inpaint the whole image.
- `mask_padding`: Padding around mask (0-256)
- `width`: Manually set inpaint area width. Leave 0 default to the masked area plus padding. (0-2048)
- `height`: Manually set inpaint area height. (0-2048)

**Outputs:**

- `inpaint_image`: Processed image for inpainting
- `inpaint_mask`: Processed mask
- `overlay_image`: Preview overlay
- `crop_region`: Crop coordinates (input of OverlayInpaintedImage)

![inpaiting prepare](https://github.com/user-attachments/assets/38e87c04-7a64-4a62-a462-054396b3de14)

#### OverlayInpaintedImage

Overlays inpainted images with original images.

**Inputs:**

- `inpainted`: Inpainted image
- `overlay_image`: Original image
- `crop_region`: Crop region coordinates

**Outputs:**

- `IMAGE`: Final composited image

#### LaMaInpaint

Remove objects from images using LaMa model.

![lama remove object](https://github.com/user-attachments/assets/c28bbd8b-d55f-4fa5-bbc9-ace267382bd0)

### LLM Nodes

- **LLM API Config**: Generic model settings (model, tokens, temperature).
- **NanoBanana API Config**: Preset for Gemini 2.5 Flash Image (modalities/aspect ratio).
- **OpenAI API**: Connect to OpenAI chat/completions.
- **OpenRouter API**: Route to many providers via OpenRouter; supports text and images where available.
- **Gemini API**: Google Gemini (text and image generation).
- **Claude API**: Anthropic Claude messages.
- **AWS Bedrock Claude API**: Claude via AWS Bedrock.
- **AWS Bedrock Mistral API**: Mistral completions via AWS Bedrock.
- **LLM Message**: Build message lists (system/user/assistant) with optional images.
- **LLM Chat**: Run multi-turn chats; returns text and optional images.
- **LLM Completion**: Single-prompt completion.

![LLM chat workflow](https://github.com/user-attachments/assets/45b8d4fd-57cd-4bd9-8274-d3e6ac4ef938)

![NanoBanana workflow](https://github.com/user-attachments/assets/9d699b47-6239-419f-b778-348618c99c4a)

# Known Issues

## AV_controlnetPreprocessor is missing

`AV_controlnetPreprocessor` is a wrapper for [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) that I created to quickly switch between multiple preprocessors, instead of having a separate node for each one. It requires `comfyui_controlnet_aux` to be installed, otherwise it will not be available.

Since `AIO_Preprocessor` is already implemented in `comfyui_controlnet_aux`, this node will be deprecated. You are recommended to switch to using `AIO_Preprocessor` node directly.

