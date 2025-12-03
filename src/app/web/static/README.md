# Web Application Structure

This directory contains the web interface for the Multimodal Classifier following SOLID principles and proper separation of concerns.

## Directory Structure

```
static/
├── index.html          # Main HTML structure (semantic markup only)
├── css/
│   └── styles.css      # All styling and visual presentation
├── js/
│   └── app.js          # All application logic and interactivity
└── README.md           # This file
```

## Architecture

### HTML (`index.html`)
- **Responsibility**: Semantic structure and content organization
- **Contains**: Pure HTML markup with proper semantic elements
- **Does NOT contain**: Inline styles or scripts (except necessary onclick handlers for simplicity)
- **Links to**: External CSS and JavaScript files

### CSS (`css/styles.css`)
- **Responsibility**: All visual presentation and styling
- **Contains**: 
  - Layout definitions (grid, flexbox)
  - Color schemes and gradients
  - Typography and spacing
  - Animations and transitions
  - Responsive design (media queries)
- **Organization**: Grouped by component with clear section headers

### JavaScript (`js/app.js`)
- **Responsibility**: All application behavior and business logic
- **Contains**:
  - API communication
  - State management
  - Event handlers
  - DOM manipulation
  - WebSocket connections
  - Image processing
- **Organization**: 
  - Clear function documentation
  - Separated by concern (Server, Image, Webcam, UI)
  - Configuration at the top
  - Initialization at the bottom

## Benefits of This Structure

1. **Maintainability**: Each file has a single, clear responsibility
2. **Scalability**: Easy to add new features without affecting other concerns
3. **Collaboration**: Multiple developers can work on different files simultaneously
4. **Performance**: CSS and JS files are cached by the browser
5. **Testing**: JavaScript logic can be tested independently
6. **Debugging**: Issues are easier to locate and fix
7. **Code Reusability**: Styles and scripts can be shared across pages

## Usage

The FastAPI server automatically serves these static files:
- Main page: `http://localhost:8000/`
- CSS: `http://localhost:8000/static/css/styles.css`
- JavaScript: `http://localhost:8000/static/js/app.js`

## Development Guidelines

### When Editing HTML:
- Focus only on structure and content
- Use semantic HTML5 elements
- Keep accessibility in mind (ARIA labels, alt text)
- No inline styles or scripts

### When Editing CSS:
- Add comments for major sections
- Use consistent naming conventions
- Group related styles together
- Test responsive behavior

### When Editing JavaScript:
- Document all functions with JSDoc comments
- Keep functions small and focused
- Use meaningful variable names
- Handle errors gracefully
- Test browser compatibility

## SOLID Principles Applied

- **Single Responsibility**: Each file handles one aspect (structure, style, or behavior)
- **Open/Closed**: Easy to extend functionality without modifying existing code
- **Liskov Substitution**: Functions are modular and can be replaced
- **Interface Segregation**: Clear API contracts between components
- **Dependency Inversion**: Loose coupling between HTML, CSS, and JS
